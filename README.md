# Oyuncu Davranışı Analitiği ve Erken Terk (Churn) Tahmin Sistemi 

Bu proje, bir mobil oyunun oyuncu verilerini uçtan uca işleyerek, "bir oyuncunun yakın zamanda oyunu terk edip etmeyeceğini" (churn) tahmin eden çalışan bir makine öğrenmesi sistemi geliştirmeyi amaçlamaktadır.

## Proje Dizin Yapısı

* `data/`: Projede kullanılan ham dosyalar ve üretilen temizlenmiş veri setleri (Kaggle/Sentetik).
* `notebooks/`: EDA (Keşifsel Veri Analizi) ve churn tahmin modellemelerinin yapıldığı Jupyter not defterleri.
* `src/`: Veri işleme ve sentetik veri üretimi için kullanılan Python scriptleri.
* `reports/`: Haftalık analiz raporları.
* `README.md`: Proje dokümantasyonu.

## Kurulum ve Çalıştırma

### 1. Ortam Hazırlığı
Proje bağımlılıklarını izole etmek için bir sanal ortam oluşturup aktif edin:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 2. Kütüphanelerin Yüklenmesi
Gerekli paketleri requirements.txt üzerinden yükleyin:
```bash
pip install -r requirements.txt
```
### 3. Veri Üretimi ve Hazırlığı
Analiz süreçlerini başlatmadan önce verileri hazırlamak için src/ klasörü altındaki scriptleri sırasıyla çalıştırabilirsiniz:

kaggle_dataset:
```bash
python src/kaggle_dataset.py
```
my_dataset:
```bash
python src/my_dataset.py
```


