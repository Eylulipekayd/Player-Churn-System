# 🎮 Oyuncu Davranışı Analitiği ve Erken Terk (Churn) Tahmin Sistemi

Bu proje, mobil oyun dünyasındaki oyuncu verilerini uçtan uca işleyerek bir oyuncunun yakın zamanda oyunu terk edip etmeyeceğini churn makine öğrenmesi modelleri ile tahmin eden ve sonuçları etkileşimli bir **Streamlit** panosunda sunan bir projedir.

---

## 📁 Proje Dizin Yapısı

```text
├── .venv/                   # Sanal ortam klasörü
├── data/                    # Ham ve pipeline'dan geçmiş temiz veriler (Kaggle & My)
├── notebooks/               # Jupyter Not Defterleri
│   ├── EDA/                 # EDA çalışmaları
│   └── models/              # Model eğitimi, test denemeleri
├── reports/                 # Haftalık analiz raporları ve çıktı görselleri
│   ├── Hafta_1/
│   ├── Hafta_2/
│   └── Hafta_3/
├── saved_models/            # Eğitilmiş ve paketlenmiş Random Forest modelleri (.pkl)
├── src/                     # Modüler kaynak kodlar
│   ├── datasets/            # Veri seti üretim scriptleri
│   ├── pipeline/            # Özellik mühendisliği ve veri temizleme hatları
│   ├── predictions/         # Model tahmin fonksiyonları ve çıkarım scriptleri
│   └── trains/              # Model eğitim, çapraz doğrulama ve metrik kaydetme scriptleri
├── tests/                   # Pytest ile veri kalitesi ve sınır durum testleri
├── pano.py                  # Streamlit dashboard ana uygulama kodu
├── requirements.txt         # Proje bağımlılıkları ve kütüphane versiyonları
└── README.md                # Proje dokümantasyonu
```

---

### 1. Ortamın Kurulması & Bağımlılıkların Yüklenmesi

**Windows için:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux için:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Veri Üretimi, Pipeline ve Test Aşaması

Verilerin üretilmesi, pipelinedan geçirilmesi ve veri kalitesi testlerinin çalıştırılması:

```bash
# Veri Setlerini Oluşturma (Hafta 1)
python src/datasets/kaggle_dataset.py
python src/datasets/my_dataset.py

# Pipeline & Özellik Mühendisliği (Hafta 2)
python src/pipeline/kaggle_pipeline.py
python src/pipeline/my_pipeline.py

# Veri Kalitesi ve Sınır Durum Testleri (Pytest)
pytest
```

---

### 3. Model Eğitimi, Kod Temizliği ve Panonun Başlatılması

Modellerin eğitilmesi ve `.pkl` olarak kaydedilmesi, kodların PEP8 standartlarına göre denetlenmesi ve canlı Streamlit panosunun başlatılması:

```bash
# Modelleri Eğitme & Raporlama (Hafta 3)
python src/trains/kaggle_train.py
python src/trains/my_train.py

# Tahmin Scriptlerini Çalıştırma
python src/predictions/kaggle_prediction.py
python src/predictions/my_prediction.py

# Kod Kalitesi ve Denetim (Clean Code)
black src/ pano.py
flake8 --ignore=E501 src/ pano.py

# Streamlit Panosunu Çalıştırma (Hafta 4)
streamlit run pano.pypython src/predictions/my_prediction.py
```

---

## 📊 Streamlit Panosu (Pano Özellikleri)

Tarayıcıda açılan panoda 4 ana sekme yer almaktadır:

1. **📊 Genel Bakış
2. **⚠️ Risk Altındaki Oyuncular
3. **📈 Model Metrikleri
4. **📊 Grafik