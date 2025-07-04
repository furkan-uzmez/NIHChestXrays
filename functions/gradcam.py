import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ---- 1. Görüntüyü hazırla ----
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

# ---- 2. Grad-CAM sınıfı ----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Grad-CAM hesaplama
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = self.activations[0] * pooled_gradients[:, None, None]
        heatmap = torch.sum(weighted_activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

# ---- 3. Görselleştirme ----
def show_cam_on_image(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# ---- 4. Kullanım ----
def get_heatmap_for_resnet(model,image_path):
    model.eval()

    # Hedef layer: resnet50 için layer4[-1]
    target_layer = model.layer4[-1]

    # Grad-CAM başlat
    cam = GradCAM(model, target_layer)

    # Görseli yükle
    input_tensor, original_image = preprocess_image(image_path)

    # Grad-CAM üret
    heatmap = cam.generate(input_tensor)

    # Görselleştir
    show_cam_on_image(original_image, heatmap)


    
