# NIH Chest X-Rays - PA/AP Görüntü Sınıflandırması

Bu proje, **NIH ChestX-ray14** veri setini kullanarak akciğer röntgenlerinde **PA (posteroanterior)** ve **AP (anteroposterior)** projeksiyon türlerini sınıflandırmayı amaçlamaktadır. Proje PyTorch kullanılarak geliştirilmiştir.

## 🧠 Model

- Kullanılan mimari: **ResNet-34**
- Transfer learning yaklaşımıyla önceden eğitilmiş ImageNet ağı üzerine ince ayar yapılmıştır.
- Çıkış katmanı, PA ve AP sınıflarını tahmin edecek şekilde 2 sınıfa ayarlanmıştır.

## 🔍 Kullanılan Veri Seti

- [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)  
- İçerik: 112,000+ röntgen görüntüsü
- Hedef: Görüntünün **PA** mı yoksa **AP** mi olduğunu tahmin etmek

