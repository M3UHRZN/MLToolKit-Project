# 🤖 ML Classification Toolkit

GUI tabanlı makine öğrenmesi sınıflandırma ve değerlendirme uygulaması.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Gereksinimler](#-gereksinimler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Modeller](#-modeller)
- [Metrikler](#-metrikler)
- [Ekran Görüntüleri](#-ekran-görüntüleri)

---

## ✨ Özellikler

### Veri İşleme
- 📂 CSV dosyası yükleme
- 🎯 Otomatik hedef sütun önerisi
- 📊 Veri seti özeti (satır/sütun sayısı, eksik değerler, sütun türleri)
- 📈 Sınıf dağılımı görüntüleme

### Ön İşleme
- 🔄 **One-Hot Encoding**: Kategorik değişkenleri sayısal forma dönüştürme
- 📏 **Normalizasyon**: StandardScaler veya MinMaxScaler ile ölçeklendirme
- 🎛️ **Eksik Değer Doldurma**: Sayısal için medyan, kategorik için en sık değer
- 🗂️ **Binning**: Sayısal hedef değişkeni sınıflara ayırma (3, 5 veya 7 sınıf)

### Model Seçenekleri
- ⚡ **Perceptron**: Hızlı lineer sınıflandırıcı
- 🧠 **MLP (Multi-Layer Perceptron)**: Yapay sinir ağı (1-4 gizli katman)
- 🌳 **Decision Tree**: Karar ağacı sınıflandırıcısı

### MLP Yapılandırması
- Gizli katman sayısı (1-4)
- Her katman için nöron sayısı
- Aktivasyon fonksiyonu (ReLU, Tanh, Logistic)
- Öğrenme oranı
- Maksimum iterasyon sayısı

### Değerlendirme
- 📊 Train/Test split oranı ayarlama (0.10 - 0.50)
- 📋 Metrik tablosu (Accuracy, Precision, Recall, F1-Score)
- 🎨 Confusion Matrix görselleştirme
- 📝 Detaylı çalıştırma günlüğü

---

## 📦 Gereksinimler

```
Python >= 3.8
pandas
numpy
scikit-learn
matplotlib
tkinter (Python ile birlikte gelir)
```

---

## 🚀 Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Projeyi Çalıştırın

```bash
python app.py
```

---

## 📖 Kullanım

### Adım 1: Veri Seti Yükleme
1. **Dataset** sekmesinde "Upload CSV" butonuna tıklayın
2. CSV dosyanızı seçin (ilk satır sütun adları olmalı)
3. Hedef (label) sütununu seçin veya "Auto-pick" ile otomatik seçim yapın

### Adım 2: Ayarları Yapılandırma
1. **Settings** sekmesine geçin
2. Ön işleme seçeneklerini ayarlayın:
   - One-Hot Encoding (kategorik özellikler için)
   - Normalizasyon (StandardScaler veya MinMaxScaler)
3. Train/Test split oranını belirleyin
4. Kullanmak istediğiniz modelleri seçin
5. MLP kullanıyorsanız, hiperparametreleri ayarlayın

### Adım 3: Eğitim ve Değerlendirme
1. "Train & Evaluate" butonuna tıklayın
2. Eğitim tamamlanana kadar bekleyin
3. **Results** sekmesinde sonuçları inceleyin

### Adım 4: Sonuçları İnceleme
- Metrik tablosunda tüm modellerin performansını karşılaştırın
- Confusion Matrix dropdown'ından model seçerek matrisi görüntüleyin
- Run Log'da detaylı bilgileri inceleyin

---

## 📁 Proje Yapısı

```
ml-project2/
├── app.py                          # Ana GUI uygulaması (Tkinter)
├── ml_core.py                      # ML mantığı (ön işleme, eğitim, değerlendirme)
├── ui_helpers.py                   # UI yardımcı fonksiyonları (ToolTip)
├── sample_classification_risk.csv  # Örnek veri seti
└── README.md                       # Bu dosya
```

### Dosya Açıklamaları

| Dosya | Açıklama |
|-------|----------|
| `app.py` | Tkinter tabanlı grafiksel kullanıcı arayüzü. Sekmeler, butonlar, grafikler ve kullanıcı etkileşimlerini yönetir. |
| `ml_core.py` | Makine öğrenmesi çekirdek mantığı. Veri ön işleme, model oluşturma, eğitim ve değerlendirme fonksiyonlarını içerir. |
| `ui_helpers.py` | Tooltip gibi UI yardımcı bileşenlerini içerir. |

---

## 🤖 Modeller

### Perceptron
- **Tür**: Tek katmanlı lineer sınıflandırıcı
- **Avantajlar**: Hızlı eğitim, basit yapı
- **Öneriler**: Normalizasyon ile daha iyi çalışır

### MLP (Multi-Layer Perceptron)
- **Tür**: Çok katmanlı yapay sinir ağı
- **Avantajlar**: Doğrusal olmayan örüntüleri öğrenebilir
- **Parametreler**:
  - `hidden_layers`: 1-4 arası gizli katman
  - `neurons`: Her katman için nöron sayısı
  - `activation`: relu, tanh, logistic
  - `learning_rate_init`: Başlangıç öğrenme oranı
  - `max_iter`: Maksimum iterasyon

### Decision Tree
- **Tür**: Karar ağacı tabanlı sınıflandırıcı
- **Avantajlar**: Yorumlanabilir, ölçeklendirme gerektirmez
- **Öneriler**: Aşırı öğrenmeye dikkat

---

## 📊 Metrikler

| Metrik | Açıklama |
|--------|----------|
| **Accuracy** | Doğru tahmin oranı (toplam doğru / toplam örnek) |
| **Precision** | Pozitif tahminlerin doğruluğu (TP / (TP + FP)) |
| **Recall** | Gerçek pozitifleri bulma oranı (TP / (TP + FN)) |
| **F1-Score** | Precision ve Recall'ın harmonik ortalaması |

> 💡 Çok sınıflı problemlerde **weighted average** kullanılır.

---

## 🖼️ Ekran Görüntüleri

### Dataset Sekmesi
- CSV yükleme
- Hedef sütun seçimi
- Veri özeti görüntüleme

### Settings Sekmesi
- Ön işleme ayarları
- Model seçimi
- MLP hiperparametreleri

### Results Sekmesi
- Metrik tablosu
- Confusion Matrix görselleştirmesi
- Çalıştırma günlüğü

---

## ⚠️ Önemli Notlar

1. **Sayısal Hedef Değişkenler**: Eğer hedef sütununuz sayısal ise (örn: yaş, gelir), 25'ten fazla benzersiz değer varsa binning önerilir.

2. **Kategorik Özellikler**: Giriş özellikleriniz kategorik veri içeriyorsa One-Hot Encoding'i aktif bırakın.

3. **MLP Yakınsama Uyarısı**: MLP modeli belirtilen iterasyon sayısında yakınsayamazsa uyarı alabilirsiniz. Bu durumda `max_iter` değerini artırabilirsiniz.

4. **Veri Kalitesi**: Eksik değerler otomatik olarak doldurulur (medyan/en sık değer).

---

## 🔧 Geliştirme

### Yeni Model Eklemek

`ml_core.py` dosyasındaki `get_models()` fonksiyonuna yeni model ekleyebilirsiniz:

```python
from sklearn.ensemble import RandomForestClassifier

def get_models(cfg: TrainConfig) -> Dict[str, object]:
    models = {}
    # ... mevcut modeller ...
    
    if cfg.use_random_forest:  # Yeni bayrak
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=100,
            random_state=cfg.random_state
        )
    
    return models
```
