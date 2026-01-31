# COVID-CXNet Evaluation Benchmark

This document summarizes the evaluation results for various NIH models evaluated on the COVID-CXNet dataset (Test Split).

## Summary Table

| Model | Status | Accuracy | AUC | F1 Weighted |
| :--- | :--- | :--- | :--- | :--- |
| xceptionfullyfinetunebestmodel.pth | Success | 0.8538 | 0.9296 | 0.8534 |
| densenetfullyfinetunebestmodel.pth | Success | 0.8405 | 0.9297 | 0.8403 |
| radimagenet_resnet50_finetuned.pth | Success | 0.5465 | 0.6005 | 0.5282 |
| 512x512resnet50nofinetunebestmodel.pth | Success | 0.6917 | 0.7644 | 0.6878 |
| resnet50nofinetunebestmodel.pth | Success | 0.7981 | 0.8964 | 0.7970 |
| resnet50fullyfinetunebestmodel.pth | Success | 0.7423 | 0.8703 | 0.7377 |
| resnet50halffinetunebestmodel.pth | Success | 0.8337 | 0.9349 | 0.8332 |
| best_model.pth (ResNet18) | Success | 0.4768 | 0.5000 | 0.3079 |
| mobilenetfullyfinetunebestmodel.pth | Success | 0.8424 | 0.9148 | 0.8424 |
| senetfullyfinetunebestmodel.pth | Success | 0.8083 | 0.9185 | 0.8074 |
| resnet_34_best_model.pth | Success | 0.7612 | 0.8774 | 0.7585 |
| resnet_34_model_weights.pth | Success | 0.7651 | 0.8741 | 0.7632 |
| simclrv1fullyfinetunebestmodel.pth | Success | 0.5232 | 0.8923 | 0.3594 |

## Evaluation Details

Full evaluation logs are available at:
[/home/furkan/Projects/NIHChestXrays/COVID-CXNet/evaluation_results_test.log](file:///home/furkan/Projects/NIHChestXrays/COVID-CXNet/evaluation_results_test.log)

> [!NOTE]
> `simclrv1fullyfinetunebestmodel.pth` shows high AUC (0.89) but low Accuracy/F1, suggesting a threshold calibration issue or class imbalance sensitivity (F1 dropped significantly).
