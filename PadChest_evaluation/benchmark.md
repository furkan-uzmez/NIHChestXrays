# PadChest Evaluation Benchmark

This document summarizes the evaluation results for various models evaluated on the PadChest dataset.

## Summary Table

| Model | Status | Accuracy | AUC | F1 Weighted |
| :--- | :--- | :--- | :--- | :--- |
| xceptionfullyfinetunebestmodel.pth | Success | 0.3196 | 0.5425 | 0.1549 |
| densenetfullyfinetunebestmodel.pth | Success | 0.3265 | 0.5503 | 0.1746 |
| radimagenet_resnet50_finetuned.pth | Success | 0.5774 | 0.5183 | 0.5746 |
| 512x512resnet50nofinetunebestmodel.pth | Success | 0.3216 | 0.5428 | 0.1628 |
| resnet50nofinetunebestmodel.pth | Success | 0.3196 | 0.4783 | 0.1549 |
| resnet50fullyfinetunebestmodel.pth | Success | 0.3304 | 0.5488 | 0.1848 |
| resnet50halffinetunebestmodel.pth | Success | 0.3374 | 0.5264 | 0.2066 |
| best_model.pth (ResNet50) | Success | 0.6804 | 0.5000 | 0.5509 |
| mobilenetfullyfinetunebestmodel.pth | Success | 0.3196 | 0.5124 | 0.1549 |
| senetfullyfinetunebestmodel.pth | Failed Load | NaN | NaN | NaN |
| best_model.pth (ResNet34) | Success | 0.3881 | 0.5480 | 0.3327 |
| model_weights.pth (ResNet34) | Success | 0.4235 | 0.5501 | 0.3984 |
| simclrv1fullyfinetunebestmodel.pth | Success | 0.3196 | 0.5225 | 0.1549 |

> [!NOTE]
> `best_model.pth` and `model_weights.pth` appear multiple times in the logs as they were evaluated from different paths or architectures (e.g., ResNet50 vs ResNet34).

## Evaluation Details

Full evaluation logs are available at:
[/home/furkan/Projects/NIHChestXrays/PadChest_evaluation/evaluation_results_full_dataset.log](file:///home/furkan/Projects/NIHChestXrays/PadChest_evaluation/evaluation_results_full_dataset.log)
