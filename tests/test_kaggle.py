import pandas as pd
import numpy as np
from src.pipeline.kaggle_pipeline import clean_data, add_features


def test_kaggle_clean_data():
    """ clean_data fonksiyonunun negatif değerleri 0 yapıp yapmadığını,
    boş satırları sildiğini ve temiz verilere dokunmadığını test eder """

    # Sahte veri oluşturulur.
    sahte_veri = pd.DataFrame({
        "OyuncuID": [1, 2, 3],
        "OturumSayisi": [10, -5, 15],
        "OyuncuSeviyesi": [50, -1, 20],
        "TotalSatınAlma": [1, -1, 1],
        "Ortalama_Oturum_Suresi": [120, -10, 80],
        "OyunaBaglilik": ["High", "Medium", np.nan]
    })

    # Fonksiyonu çalıştırır.
    temizlenmis_df = clean_data(sahte_veri)

    # Kontrollerin yapıldığı kısımdır.
    # dropna() yüzünden np.nan içeren 3. satır tamamen silindiği için geriye 2 satır kalmalıdır.
    assert len(temizlenmis_df) == 2

    # Negatif değerlerin başarıyla 0'a dönüştürüldüğünü doğruluyoruz
    assert temizlenmis_df.loc[1, "OturumSayisi"] == 0
    assert temizlenmis_df.loc[1, "OyuncuSeviyesi"] == 0
    assert temizlenmis_df.loc[1, "TotalSatınAlma"] == 0
    assert temizlenmis_df.loc[1, "Ortalama_Oturum_Suresi"] == 0

    #  Temiz oyuncunun hiçbir sütununa yanlışlıkla zarar verilmediğini doğruluyoruz.
    assert temizlenmis_df.loc[0, "OturumSayisi"] == 10
    assert temizlenmis_df.loc[0, "OyuncuSeviyesi"] == 50
    assert temizlenmis_df.loc[0, "TotalSatınAlma"] == 1
    assert temizlenmis_df.loc[0, "Ortalama_Oturum_Suresi"] == 120
    assert temizlenmis_df.loc[0, "OyunaBaglilik"] == "High"


def test_kaggle_add_features():
    """ add_features fonksiyonunun yeni sütunları doğru hesaplayıp hesaplamadığını test eder """

    # İkinci satırın oturum sayısını 0 yapıyoruz ki sıfıra bölme hatasını (ZeroDivisionError) test eder.
    sahte_veri = pd.DataFrame({
        "OturumSayisi": [5, 0],
        "Ortalama_Oturum_Suresi": [10, 20],
        "TotalSatınAlma": [10, 0]
    })

    # Fonksiyonu çalıştırır.
    yeni_df = add_features(sahte_veri)

    # Kontrollerin yapıldığı kısımdır.
    # Toplam_Oyun_Suresi doğru çarpılmış mı? (5 * 10 = 50 olmalı)
    assert yeni_df.loc[0, "Toplam_Oyun_Suresi"] == 50

    # Oturum_Basina_Harcama doğru bölünmüş mü? (10 / 5 = 2.0 olmalı)
    assert yeni_df.loc[0, "Oturum_Basina_Harcama"] == 2.0

    # Oturum sayısı 0 olan satırda np.where sayesinde hata almayıp 0 yazdığını doğruluyor.
    assert yeni_df.loc[1, "Oturum_Basina_Harcama"] == 0