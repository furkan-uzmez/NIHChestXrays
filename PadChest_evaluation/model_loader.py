import torch
import torch.nn as nn
from torchvision import models
import timm
import logging
import os

def strip_prefix(state_dict):
    """Strip common prefixes like 'module.', 'model.', '_orig_mod.' from state dict keys."""
    prefixes = ['module.', 'model.', '_orig_mod.']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
        new_state_dict[name] = v
    return new_state_dict


def get_num_classes_from_state_dict(state_dict):
    """Detect number of classes in the classification head."""
    # Common head names in torchvision and timm
    candidates = [
        'fc.weight', 'classifier.weight', 'classifier.6.weight', 
        'head.fc.weight', 'head.weight', 'fc.bias', 'classifier.bias',
        'conv_head.weight', 'conv_head.bias'
    ]
    
    # First check exact matches
    for key in candidates:
        if key in state_dict:
            return state_dict[key].shape[0]
            
    # Then check for names containing these strings
    for key in state_dict.keys():
        if any(c in key for c in ['fc.weight', 'classifier.weight', 'head.weight']):
            if len(state_dict[key].shape) == 2:
                return state_dict[key].shape[0]

    # Fallback to last weight layer
    for key in reversed(list(state_dict.keys())):
        if 'weight' in key and len(state_dict[key].shape) == 2:
            return state_dict[key].shape[0]
    return 2


def replace_head(model, num_classes):
    """Replace the classification head of the model."""
    if isinstance(model, models.ResNet):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif isinstance(model, models.DenseNet):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif isinstance(model, models.MobileNetV2):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        else:
            logging.warning("  Could not automatically replace head.")
    return model


def try_load_architecture(arch_name, state_dict, pretrained_classes):
    """Attempt to load state_dict into a specific architecture."""
    # Strip prefixes first
    state_dict = strip_prefix(state_dict)
    
    try:
        model = None
        # Try torchvision architectures
        if arch_name == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, pretrained_classes)
        elif arch_name == 'resnet34':
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, pretrained_classes)
        elif arch_name == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, pretrained_classes)
        elif arch_name == 'densenet121':
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, pretrained_classes)
        elif arch_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, pretrained_classes)
        
        # Try loading with strict=False to handle minor discrepancies
        if model is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logging.info(f"    {arch_name} (torchvision) missing: {len(missing)}, unexpected: {len(unexpected)}")
            # If too many missing keys, maybe it's not the right architecture (or it's from timm)
            if len(missing) < 100: # Slightly relaxed threshold
                return model
        
        # Try timm as fallback or for specific architectures
        timm_archs = {
            'xception': ['xception'],
            'seresnet50': ['seresnet50'],
            'resnet50': ['resnet50'],
            'mobilenet_v2': ['mobilenetv2_100', 'mobilenetv3_large_100', 'mobilenetv3_small_100'],
            'densenet121': ['densenet121']
        }
        
        if arch_name in timm_archs or model is None:
            candidates = timm_archs.get(arch_name, [arch_name])
            for t_arch in candidates:
                try:
                    model = timm.create_model(t_arch, pretrained=False, num_classes=pretrained_classes)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    logging.info(f"    {t_arch} (timm) missing: {len(missing)}, unexpected: {len(unexpected)}")
                    if len(missing) < 100:
                        return model
                except Exception as e:
                    logging.info(f"    Timm error for {t_arch}: {e}")
                    pass
                
        return None
    except Exception as e:
        logging.info(f"    Failed {arch_name} load attempt: {e}")
        return None


def load_model_and_adjust(model_path, target_classes, device):
    """High-level function to load a model checkpoint and adjust its head."""
    model_name = os.path.basename(model_path).lower()
    
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        logging.error(f"  Failed to load state dict: {e}")
        return None
        
    pretrained_classes = get_num_classes_from_state_dict(state_dict)
    logging.info(f"  Checkpoint has {pretrained_classes} classes.")
    
    arch_candidates = []
    if 'resnet18' in model_name:
        arch_candidates = ['resnet18', 'resnet34', 'resnet50']
    elif 'resnet34' in model_name:
        arch_candidates = ['resnet34', 'resnet18', 'resnet50']
    elif 'resnet50' in model_name or 'radimagenet' in model_name or 'simclr' in model_name:
        arch_candidates = ['resnet50', 'resnet34', 'seresnet50']
    elif 'densenet' in model_name:
        arch_candidates = ['densenet121']
    elif 'mobilenet' in model_name:
        arch_candidates = ['mobilenet_v2']
    elif 'xception' in model_name:
        arch_candidates = ['xception']
    elif 'senet' in model_name:
        arch_candidates = ['senet154', 'seresnet50', 'seresnext50_32x4d', 'resnet50']
    else:
        arch_candidates = ['resnet50', 'resnet34', 'resnet18']

    model = None
    for arch in arch_candidates:
        model = try_load_architecture(arch, state_dict, pretrained_classes)
        if model is not None:
            logging.info(f"  Successfully loaded weights into {arch}")
            break
            
    if model is None:
        logging.error(f"  Could not match architecture. Candidates tried: {arch_candidates}")
        return None
        
    if pretrained_classes != target_classes:
        logging.info(f"  Adjusting head from {pretrained_classes} to {target_classes} classes...")
        model = replace_head(model, target_classes)
        
    model = model.to(device)
    return model
