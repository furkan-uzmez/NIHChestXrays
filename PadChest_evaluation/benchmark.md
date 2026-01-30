# PadChest Evaluation Benchmark

This document summarizes the evaluation results for various models evaluated on the PadChest dataset.

## Summary Table

| Model | Status | Accuracy | AUC | F1 Weighted | TPR | FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| xceptionfullyfinetunebestmodel.pth | Success | 0.9854 | 0.9525 | 0.9866 | 0.9899 | 0.3571 |
| densenetfullyfinetunebestmodel.pth | Success | 0.9935 | 0.9104 | 0.9930 | 0.9986 | 0.3929 |
| radimagenet_resnet50_finetuned.pth | Success | 0.7320 | 0.6201 | 0.8339 | 0.7367 | 0.6250 |
| 512x512resnet50nofinetunebestmodel.pth | Success | 0.8055 | 0.8174 | 0.8807 | 0.8071 | 0.3214 |
| resnet50nofinetunebestmodel.pth | Success | 0.9891 | 0.9040 | 0.9889 | 0.9951 | 0.4643 |
| resnet50fullyfinetunebestmodel.pth | Success | 0.9768 | 0.8930 | 0.9803 | 0.9824 | 0.4464 |
| resnet50halffinetunebestmodel.pth | Success | 0.9884 | 0.9497 | 0.9887 | 0.9934 | 0.3929 |
| best_model.pth (ResNet50) | Success | 0.9870 | 0.5000 | 0.9806 | 1.0000 | 1.0000 |
| mobilenetfullyfinetunebestmodel.pth | Success | 0.9444 | 0.8569 | 0.9614 | 0.9488 | 0.3929 |
| senetfullyfinetunebestmodel.pth | Success (Load Fixed) | Pending | Pending | Pending | Pending | Pending |
| best_model.pth (ResNet34) | Success | 0.9898 | 0.9271 | 0.9890 | 0.9967 | 0.5357 |
| model_weights.pth (ResNet34) | Success | 0.9877 | 0.9254 | 0.9874 | 0.9944 | 0.5179 |
| simclrv1fullyfinetunebestmodel.pth | Success | 0.0130 | 0.9321 | 0.0003 | 0.0000 | 0.0000 |

> [!NOTE]
> `best_model.pth` and `model_weights.pth` appear multiple times in the logs as they were evaluated from different paths or architectures (e.g., ResNet50 vs ResNet34).

## Evaluation Details

Full evaluation logs are available at:
[/home/furkan/Projects/NIHChestXrays/PadChest_evaluation/evaluation_results_full_dataset.log](file:///home/furkan/Projects/NIHChestXrays/PadChest_evaluation/evaluation_results_full_dataset.log)

> [!NOTE]
> `senetfullyfinetunebestmodel.pth` was identified as `seresnext50_32x4d` and the loader has been updated to support it. Evaluation was started but terminated early; full results are pending execution of the full evaluation script.
