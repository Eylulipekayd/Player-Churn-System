import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def oyuncu_profili_uret(n_players=10000):
    """
        Rastgele oyuncu profilleri oluşturarak sentetik bir veri seti üretir.
        Oyuncuların yaşı, kayıt tarihi, giriş alışkanlıkları ve churn (terk etme) durumlarını simüle eder.
        """
    # Sonuçların her seferinde aynı olması için sabit bir değer var
    # 42'nin özel bir anlamı yok yazılımda yaygın bir sayı olduğu için tercih edildi
    np.random.seed(42)
    bugun = datetime.now()
    oyuncu_verisi = []

    for i in range(n_players):
        #Kod her seferinde rastgele bir yaş üretir ama bu yaşlar genellikle 20'ye yakın çıkar.
        # Standart sapma yani yaşların 20'den ne kadar uzaklaşabileceği 15 dir böyelece 20-35 yaş olasılığı daha fazladır.
        yas = int(np.random.normal(20, 15))

        #60 dan büyük gelen sayıları 60 a indir 12 den küçük gelen sayıları 12 ye çıkar
        yas = max(12, min(yas, 60))

        #12 yaşından küçüklerin oyuna girmiş olmamasını sağlıyor.
        max_oyun_suresi_gun = (yas - 12) * 365

        #Oyuncunun sisteme kayıt olabileceği en eski günü 1 ila 1095 gün arasına sabitler (Oyun 3 yıllık)
        ust_sinir = max(1, min(1095, max_oyun_suresi_gun))

        #Random gün farkı değeriyle oyuncuya bir kayıt olduğu tarih hesaplar.
        gun_farki = np.random.randint(0, ust_sinir)
        kayit_tarihi = bugun - timedelta(days=gun_farki)

        #hesap yaşı için bir hesaplama
        toplam_gun = (bugun - kayit_tarihi).days

        # Veri setindeki aktif/pasif oyuncu oranını dengelemek için %50 %50 ayarlar.
        if np.random.random() < 0.5:
            aktiflik_esigi = min(180, toplam_gun)
            gun_farki_son_giris = toplam_gun - np.random.randint(0, aktiflik_esigi + 1)
        else:
            gun_farki_son_giris = np.random.randint(0, toplam_gun + 1)

        son_giris = kayit_tarihi + timedelta(days=gun_farki_son_giris)

        # 12-22 yaş aralığına öğrenci etiketi verir.
        is_ogrenci = 12 <= yas <= 22
        mevsimsel_saatler = {"Ilkbahar": 0, "Yaz": 0, "Sonbahar": 0, "Kış": 0}


        current = kayit_tarihi
        while current <= son_giris:
            ay = current.month
            # Tarihe göre mevsimsel sınıflandırma yapar.
            mevsim = "Ilkbahar" if ay in [3, 4, 5] else (
                "Yaz" if ay in [6, 7, 8] else ("Sonbahar" if ay in [9, 10, 11] else "Kış"))

            #Yaz mevsiminde oyun süresi çarpanı daha fazla olur.
            carpan = 3.5 if (mevsim == "Yaz" and is_ogrenci) else (1.2 if mevsim == "Yaz" else 1.0)
            gunluk_saat = np.random.normal(0.5 * carpan, 0.1)
            mevsimsel_saatler[mevsim] += max(0, gunluk_saat)
            current += timedelta(days=1)

        # Bugün ile son giriş tarihi arasındaki farkı hesaplar
        gun_farki_guncel = (bugun - son_giris).days
        # 21 gün ve üzeri giriş yapmayan oyuncular churn 1 olur
        is_churn = 1 if gun_farki_guncel > 21 else 0

        # Mevsimsel oyun sürelerini toplar ve genel metrikleri hesaplar
        toplam_saat = sum(mevsimsel_saatler.values())
        # Oyuncu seviyesini kare kök fonksiyonu kullanarak türetir. Saat arttıkça seviye atlama zorlaşır.
        seviye = max(1, min(100, int(np.sqrt(toplam_saat) * 2)))
        # Hesap yaşını yıl cinsinden hesaplar
        hesap_yasi = max(0.1, round((bugun - kayit_tarihi).days / 365.25, 2))

        # Hesaplanan tüm metrikleri bir sözlük yapısında birleştirir
        oyuncu_verisi.append({
            "OyuncuID": 1000 + i,
            "Yas": yas,
            "Kayit_Tarihi": kayit_tarihi.strftime("%Y-%m-%d"),
            "Son_Giris": son_giris.strftime("%Y-%m-%d"),
            "Ilkbahar_Saati": int(mevsimsel_saatler["Ilkbahar"]),
            "Yaz_Saati": int(mevsimsel_saatler["Yaz"]),
            "Sonbahar_Saati": int(mevsimsel_saatler["Sonbahar"]),
            "Kis_Saati": int(mevsimsel_saatler["Kış"]),
            "Toplam_Saat": int(toplam_saat),
            "OyuncuSeviyesi": seviye,
            "Churn_Durumu": is_churn,
            "Hesap_Yasi": hesap_yasi
        })
    # Oluşturulan tüm oyuncu kayıtlarını Pandas DataFrame formatına dönüştürür
    return pd.DataFrame(oyuncu_verisi)

# Proje dizinini dinamik olarak belirle ve 'data' klasörünün varlığını kontrol et
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # Simülasyonu başlat ve10.000 adet sentetik oyuncu verisi üret
    df = oyuncu_profili_uret(10000)
    # Üretilen veriyi 'my_dataset.csv' ismiyle CSV formatında diske kaydeder, 'index=False' parametresi ile gereksiz satır numaralarını dosyaya yazdırmaz
    df.to_csv(os.path.join(data_dir, "my_dataset.csv"), index=False)