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
python src/datasets/kaggle_dataset.py
```
my_dataset:
```bash
python src/datasets/my_dataset.py
```

## Hafta 2 Güncellemeleri: Data Pipeline & Test Altyapısı

Projenin ikinci haftasında, analiz adımları modüler hale getirilmiş ve sistem `pytest` ile test edilmiştir.

### 1. Yenilenen Klasör ve Dosya Yapısı
Proje dizinine pipeline ve testler için yeni klasörler eklenmiştir:

* `src/pipeline/`: Modüler veri işleme hatları.
    * `kaggle_pipeline.py`: Kaggle veri seti için pipeline scripti.
    * `my_pipeline.py`: Kendi ürettiğim veri seti için pipeline scripti.
  
* `tests/`: Kod kalitesini ve veri tutarlılığını denetleyen testler.
    * `test_kaggle.py`: Kaggle pipeline için veri kalitesi kontrolü yapar.
    * `test_my_dataset.py`: My pipeline için veri kalitesi kontrolü yapar.

### 2. Pipeline Terminalden Çalıştırılması
Ham verileri temizlemek ve modellemeye hazır yeni özellikleri türetmek için veri hatları terminalden tek komutla çalıştırılabilir:
```bash
python src/pipeline/kaggle_pipeline.py
```
```bash
python src/pipeline/my_pipeline.py
```
### 3.Testlerin Terminalden Çalıştırılması
Yazılan tüm veri kalitesi kontrollerini ve sınır durum testlerini çalıştırmak için terminale şu komut yazılır:
```bash
pytest
```
## Hafta 3 Güncellemeleri: Modelleme ve Train Altyapısı

Projenin üçüncü haftasında, veri işleme hatlarından gelen temizlenmiş veriler kullanılarak makine öğrenmesi modelleri eğitilmiş, model kararlılıkları çapraz doğrulama ile test edilmiş ve kapsamlı performans raporları üretilmiştir.

### 1. Yenilenen Klasör ve Dosya Yapısı
Modelleme süreçleri için projeye yeni notebooklar ve model eğitim scriptleri eklenmiştir:

* `notebooks/`
    * `models.ipynb`: Modellerin deneysel olarak kurulduğu, eğitildiği ve analiz edildiği Jupyter not defteri.

* `src/trains/`
    * `kaggle_train.py`: Kaggle veri seti üzerinde modelleri eğitir.
    * `my_train.py`: Kendi ürettiğim veri seti üzerinde modelleri eğitir.

### 2. Model Eğitim Scriptlerinin Terminalden Çalıştırılması
Hazırlanan modüler eğitim scriptlerini modelleri eğitmek, metrik tablolarını basmak ve performans grafiklerini otomatik olarak üretmek için terminalden şu komutlar çalıştırılır:

Kaggle modellerini eğitmek ve değerlendirmek için:
```bash
python src/trains/kaggle_train.py
```
Kendi veri setimin modellerini eğitmek ve değerlendirmek için:
```bash
python src/trains/my_train.py
```
### 3. Model Çıktıları
Eğitim scriptleri başarıyla çalıştıktan sonra elde edilen Precision, Recall, F1-Score, Accuracy ve Cross-Validation sonuçları otomatik olarak hesaplanır. 
Sistem, üretilen görsel çıktıları doğrudan reports/ klasörü altına kaydedecek şekilde optimize edilmiştir.