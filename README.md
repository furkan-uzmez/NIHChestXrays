# NIH Chest X-Rays - PA/AP Görüntü Sınıflandırması

Bu proje, **NIH ChestX-ray14** veri setini kullanarak akciğer röntgenlerinde **PA (posteroanterior)** ve **AP (anteroposterior)** projeksiyon türlerini sınıflandırmayı amaçlamaktadır. Proje PyTorch kullanılarak geliştirilmiştir.

## 🧠 Model

- Kullanılan mimari: **ResNet-34**
- Transfer learning yaklaşımıyla önceden eğitilmiş ImageNet ağı üzerine ince ayar yapılmıştır.
- Çıkış katmanı, PA ve AP sınıflarını tahmin edecek şekilde 2 sınıfa ayarlanmıştır.

## 🏆 Top 5 Performing Models (by Validation Loss)

Aşağıda, en düşük doğrulama kaybına (validation loss) göre sıralanmış en iyi 5 model bulunmaktadır:

| Rank | Model | Validation Accuracy | Validation Loss |
|------|-------|---------------------|-----------------|
| 1 | **Xception** | 99.61% | 0.0184 |
| 2 | **Densenet** | 99.61% | 0.0187 |
| 3 | **Mobilenet** | 99.60% | 0.0200 |
| 4 | **SimCLR ResNet50** | 99.50% | 0.0205 |
| 5 | **SeNet** | 99.58% | 0.0206 |

> **Note**: While SeNet has higher accuracy than SimCLR ResNet50, SimCLR achieved a slightly lower validation loss (0.0205 vs 0.0206), indicating high confidence in its predictions.

## 🔍 Kullanılan Veri Seti

- [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)  
- İçerik: 112,000+ röntgen görüntüsü
- Hedef: Görüntünün **PA** mı yoksa **AP** mi olduğunu tahmin etmek
