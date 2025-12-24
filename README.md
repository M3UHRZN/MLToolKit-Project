# 🤖 ML Classification Toolkit

A GUI-based machine learning classification and evaluation application.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Metrics](#-metrics)

---

## ✨ Features

### Data Processing
- 📂 CSV file upload
- 🎯 Automatic target column recommendation
- 📊 Dataset summary (rows/columns, missing values, column types)
- 📈 Class distribution visualization

### Preprocessing
- 🔄 **One-Hot Encoding**: Convert categorical variables to numerical form
- 📏 **Normalization**: Scaling with StandardScaler or MinMaxScaler
- 🎛️ **Missing Value Imputation**: Median for numeric, most frequent for categorical
- 🗂️ **Binning**: Discretize numeric target into classes (3, 5, or 7 classes)

### Model Options
- ⚡ **Perceptron**: Fast linear classifier
- 🧠 **MLP (Multi-Layer Perceptron)**: Neural network (1-4 hidden layers)
- 🌳 **Decision Tree**: Decision tree classifier

### MLP Configuration
- Number of hidden layers (1-4)
- Neurons per layer
- Activation function (ReLU, Tanh, Logistic)
- Learning rate
- Maximum iterations

### Evaluation
- 📊 Adjustable Train/Test split ratio (0.10 - 0.50)
- 📋 Metrics table (Accuracy, Precision, Recall, F1-Score)
- 🎨 Confusion Matrix visualization
- 📝 Detailed run log

---

## 📦 Requirements

```
Python >= 3.8
pandas
numpy
scikit-learn
matplotlib
tkinter (comes with Python)
```

---

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Run the Project

```bash
python app.py
```

---

## 📖 Usage

### Step 1: Load Dataset
1. Click "Upload CSV" button in the **Dataset** tab
2. Select your CSV file (first row should contain column names)
3. Choose the target (label) column or use "Auto-pick" for automatic selection

### Step 2: Configure Settings
1. Switch to the **Settings** tab
2. Configure preprocessing options:
   - One-Hot Encoding (for categorical features)
   - Normalization (StandardScaler or MinMaxScaler)
3. Set the Train/Test split ratio
4. Select the models you want to use
5. If using MLP, adjust the hyperparameters

### Step 3: Train and Evaluate
1. Click the "Train & Evaluate" button
2. Wait for training to complete
3. Review results in the **Results** tab

### Step 4: Review Results
- Compare performance of all models in the metrics table
- Select a model from the Confusion Matrix dropdown to view the matrix
- Review detailed information in the Run Log

---

## 📁 Project Structure

```
ml-project2/
├── app.py                          # Main GUI application (Tkinter)
├── ml_core.py                      # ML logic (preprocessing, training, evaluation)
├── ui_helpers.py                   # UI helper functions (ToolTip)
├── sample_classification_risk.csv  # Sample dataset
└── README.md                       # This file
```

### File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Tkinter-based graphical user interface. Manages tabs, buttons, charts, and user interactions. |
| `ml_core.py` | Machine learning core logic. Contains data preprocessing, model creation, training, and evaluation functions. |
| `ui_helpers.py` | Contains UI helper components like tooltips. |

---

## 🤖 Models

### Perceptron
- **Type**: Single-layer linear classifier
- **Advantages**: Fast training, simple structure
- **Recommendations**: Works better with normalization

### MLP (Multi-Layer Perceptron)
- **Type**: Multi-layer artificial neural network
- **Advantages**: Can learn non-linear patterns
- **Parameters**:
  - `hidden_layers`: 1-4 hidden layers
  - `neurons`: Number of neurons per layer
  - `activation`: relu, tanh, logistic
  - `learning_rate_init`: Initial learning rate
  - `max_iter`: Maximum iterations

### Decision Tree
- **Type**: Decision tree-based classifier
- **Advantages**: Interpretable, no scaling required
- **Recommendations**: Watch out for overfitting

---

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correct prediction ratio (total correct / total samples) |
| **Precision** | Accuracy of positive predictions (TP / (TP + FP)) |
| **Recall** | True positive detection rate (TP / (TP + FN)) |
| **F1-Score** | Harmonic mean of Precision and Recall |

> 💡 **Weighted average** is used for multi-class problems.

---

## ⚠️ Important Notes

1. **Numeric Target Variables**: If your target column is numeric (e.g., age, income), binning is recommended when there are more than 25 unique values.

2. **Categorical Features**: Keep One-Hot Encoding enabled if your input features contain categorical data.

3. **MLP Convergence Warning**: If the MLP model doesn't converge within the specified iterations, you may receive a warning. In this case, you can increase the `max_iter` value.

4. **Data Quality**: Missing values are automatically imputed (median/most frequent).

---
---

# 🤖 ML Sınıflandırma Araç Kiti

GUI tabanlı makine öğrenmesi sınıflandırma ve değerlendirme uygulaması.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Gereksinimler](#-gereksinimler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Modeller](#-modeller)
- [Metrikler](#-metrikler)

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

## ⚠️ Önemli Notlar

1. **Sayısal Hedef Değişkenler**: Eğer hedef sütununuz sayısal ise (örn: yaş, gelir), 25'ten fazla benzersiz değer varsa binning önerilir.

2. **Kategorik Özellikler**: Giriş özellikleriniz kategorik veri içeriyorsa One-Hot Encoding'i aktif bırakın.

3. **MLP Yakınsama Uyarısı**: MLP modeli belirtilen iterasyon sayısında yakınsayamazsa uyarı alabilirsiniz. Bu durumda `max_iter` değerini artırabilirsiniz.

4. **Veri Kalitesi**: Eksik değerler otomatik olarak doldurulur (medyan/en sık değer).
