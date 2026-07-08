import pandas as pd
import numpy as np
from src.pipeline.my_pipeline import clean_data, add_features


def test_my_data_clean_data():
    """clean_data fonksiyonunun 1000'den küçük ID'leri ve diğer negatif değerleri
    0 yapıp yapmadığını, temiz verilere ise dokunmadığını test eder
    """

    # Sahte veri oluşturulur.
    # 1. oyuncu tamamen temiz, 2. oyuncu hatalı ID (999) ve negatif tuzaklar içeriyor, 3. satır ise NaN içeriyor
    sahte_veri = pd.DataFrame(
        {
            "OyuncuID": [1001, 999, -1002],  # 999 ve -1002 -> 0 OLMALI!
            "Dogum_Tarihi": ["2002-05-10", "1995-04-12", "2005-08-20"],
            "Yas": [24, -30, 21],  # -30 -> 0 OLMALI
            "Kayit_Tarihi": ["2024-01-01", "2025-02-15", "2026-03-20"],
            "Son_Giris": ["2026-07-01", "2026-06-15", "2026-07-05"],
            "OyuncuSeviyesi": [50, -5, 20],  # -5 -> 0 OLMALI
            "Hesap_Yasi": [2.1, 1.0, 0.5],
            "Toplam_Saat": [150, 20, 80],
            "Ilkbahar_Saati": [30, 10, 25],
            "Yaz_Saati": [40, 15, 30],
            "Sonbahar_Saati": [20, 5, 15],
            "Kis_Saati": [50, 20, 45],
            "Churn_Durumu": [0, 1, np.nan],  # NaN içeren bu 3. satır komple silinecek
        }
    )

    # Fonksiyonu çalıştırır.
    temizlenmis_df = clean_data(sahte_veri)

    # Kontrollerin yapıldığı kısımdır.
    # dropna() sayesinde Churn_Durumu NaN olan satır silindi, geriye 2 satır kalmalıdır.
    assert len(temizlenmis_df) == 2

    # 1000'den küçük olan 999 ID'sinin ve negatif sayıların başarıyla 0 yapıldığını doğrular.
    assert temizlenmis_df.loc[1, "OyuncuID"] == 0
    assert temizlenmis_df.loc[1, "Yas"] == 0
    assert temizlenmis_df.loc[1, "OyuncuSeviyesi"] == 0


    # Temiz oyuncunun tüm profil sütunlarının aynen korunduğunu doğrular.
    assert temizlenmis_df.loc[0, "OyuncuID"] == 1001
    assert temizlenmis_df.loc[0, "Dogum_Tarihi"] == "2002-05-10"
    assert temizlenmis_df.loc[0, "Yas"] == 24
    assert temizlenmis_df.loc[0, "Kayit_Tarihi"] == "2024-01-01"
    assert temizlenmis_df.loc[0, "Son_Giris"] == "2026-07-01"
    assert temizlenmis_df.loc[0, "OyuncuSeviyesi"] == 50
    assert temizlenmis_df.loc[0, "Hesap_Yasi"] == 2.1
    assert temizlenmis_df.loc[0, "Toplam_Saat"] == 150
    assert temizlenmis_df.loc[0, "Ilkbahar_Saati"] == 30
    assert temizlenmis_df.loc[0, "Yaz_Saati"] == 40
    assert temizlenmis_df.loc[0, "Sonbahar_Saati"] == 20
    assert temizlenmis_df.loc[0, "Kis_Saati"] == 50
    assert temizlenmis_df.loc[0, "Churn_Durumu"] == 0


def test_my_data_add_features():
    """add_features fonksiyonunun Ilerleme_Hizi ve Oturum_Sikligi sütunlarını
    doğru hesaplayıp hesaplamadığını test eder
    """


    # İkinci satırda Hesap_Yasi'nı 0 yapıyoruz ki sıfıra bölme hatasını (ZeroDivisionError) test etsin.
    sahte_veri = pd.DataFrame(
        {
            "OyuncuSeviyesi": [6, 10],
            "Hesap_Yasi": [2.0, 0.0],
            "Toplam_Saat": [730.5, 50.0],
        }
    )

    # Fonksiyonu çalıştırır.
    yeni_df = add_features(sahte_veri)

    # Kontrollerin yapıldığı kısımdır.
    # Ilerleme_Hizi = 6 / 2.0 = 3.0 olmalıdır.
    assert yeni_df.loc[0, "Ilerleme_Hizi"] == 3.0

    # toplam_gun = 2.0 * 365.25 = 730.5 -> Oturum_Sikligi = 730.5 / 730.5 = 1.0 olmalıdır.
    assert yeni_df.loc[0, "Oturum_Sikligi"] == 1.0

    # Hesap_Yasi = 0 olan uç durumda hata almayıp 0 yazdığını doğruluyor.
    assert yeni_df.loc[1, "Oturum_Sikligi"] == 0