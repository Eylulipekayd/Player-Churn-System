import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def oyuncu_profili_uret(n_players=10000):
    """
    Rastgele oyuncu profilleri oluşturarak sentetik bir veri seti üretir.
    Oyuncunun doğum tarihi, kayıt tarihi ve bugünkü yaşını mantıksal tutarlılıkla simüle eder.
    """
    np.random.seed(42)
    bugun = datetime.now()
    oyuncu_verisi = []

    for i in range(n_players):
        #hedef yaş standart sapma mantığıyla 20 yaş ve 35 yaş aralığında daha yoğundur.
        #hedef yaş min 12 max 60 olabilir şeklinde ayarlanmıştır.
        hedef_yas = int(np.random.normal(20, 15))
        hedef_yas = max(12, min(hedef_yas, 60))

        # Hedef yaşa uygun olarak geçmişten rastgele bir doğum tarihi seçiyoruz
        # Hedef_yas=12 olan biri için toplam_gun_yas= 12 yıl 4 ay olabilir
        # Buna göre doğum tarihi de 06.03.2014 olur.
        toplam_gun_yas = (hedef_yas * 365) + np.random.randint(0, 365)
        dogum_tarihi = bugun - timedelta(days=toplam_gun_yas)


        # Oyuncu en erken 12. yaş gününde oyuna kaydolabilir.
        #en_erken_kayit_tarihi= 06.03.2026 12 yaşına geldiği tarih olur.
        en_erken_kayit_tarihi = dogum_tarihi + timedelta(days=12 * 365)

        # Oyun 3 yıl önce çıkmış olarak ele aldım.
        oyun_acilis_tarihi = bugun - timedelta(days=1095)

        # Oyuncunun kaydolabileceği tarih aralığıdır.
        # baslangic_tarihi = max(06.03.2026,06.07.2023) büyük değer seçilir.
        baslangic_tarihi = max(en_erken_kayit_tarihi, oyun_acilis_tarihi)

        #Oyuncunun potansiyel olarak kaydolabileceği en eski gün ile bugün arasındaki toplam süredir
        #kalan_gun_araligi = (06.07.2026-06.03.2026)'dan 120 gün gelir.
        kalan_gun_araligi = (bugun - baslangic_tarihi).days

        #kalan_gun_araligi de çıkan değere göre kayıt tarihi bulunur.
        if kalan_gun_araligi <= 0:
            kayit_tarihi = bugun

        # Kayıt olma tarihleri üstel olsun eşit oranlarda dağılmasın.
        # 120 için else çalışır. carpan 01*01=0,01 olsun dersek mesela gun_farki = 1,2 gündür, virgülden sonrası silinir 1 gündür.
        # Buna göre kayit_tarihi = 07.03.2026 olur.
        else:
            carpan = np.random.random() ** 2
            gun_farki = int(carpan * kalan_gun_araligi)

            kayit_tarihi = baslangic_tarihi + timedelta(days=gun_farki)


        # Artık doğum tarihi net olduğu için bugünkü yaşı küsuratlı hesaplayabiliriz
        # yas=(06.07.2026 - 06.03.2014)/ 365.25 den 12 gelir.
        yas = int((bugun - dogum_tarihi).days / 365.25)
        # toplam_gun = (06.07.2026 - 07.03.2026) den yaklaşık 120 bulunur.
        toplam_gun = (bugun - kayit_tarihi).days

        # %50 ihtimalle aktifse toplam gün ile (yaklaşık 120) 21 arasından min olan seçilir yani 21
        # gun_farki_bugunden = (0.21) arasından mesela 14 gelsin diyelim.
        # Buna göre son_giris = (06.07.2026 - 14 gün) den 22.06.2026 gelir

        if np.random.random() < 0.5:
            gun_farki_bugunden = np.random.randint(0, min(21, toplam_gun + 1))
            son_giris = bugun - timedelta(days=gun_farki_bugunden)

        #%50 ihtimalle pasif kısmında toplam gün 21 den yüksekse ki yaklaşık 120 olarak ele aldık.
        # gun_farki_bugunden =(22,121) den 80 gelebilir mesela
        # Buna göre son_giris =(06.07.2026 - 80 gün) den 17.04.2026 olur.
        else:

            if toplam_gun > 21:
                gun_farki_bugunden = np.random.randint(22, toplam_gun + 1)
            else:
                gun_farki_bugunden = toplam_gun

            son_giris = bugun - timedelta(days=gun_farki_bugunden)

        # girilmeyen gün sayısına göre churn hesabı yapılır 21 ve sonrası churn durumundadır.
        girilmeyen_gun_sayisi = (bugun - son_giris).days
        churn = 1 if girilmeyen_gun_sayisi > 21 else 0

        # öğrenci yaş grubunu bulur.
        ogrenci_mi = 12 <= yas <= 22
        mevsimsel_saatler = {"Ilkbahar": 0, "Yaz": 0, "Sonbahar": 0, "Kış": 0}

        # yaz mevsiminde öğrenci oyun saati çarpanı artar.
        # Günlük oyun saati mevsimlere göre hesaplanır ve sözlüğe eklenir.
        current = kayit_tarihi
        while current <= son_giris:
            ay = current.month
            mevsim = "Ilkbahar" if ay in [3, 4, 5] else (
                "Yaz" if ay in [6, 7, 8] else ("Sonbahar" if ay in [9, 10, 11] else "Kış"))

            carpan = 3.5 if (mevsim == "Yaz" and ogrenci_mi) else (1.2 if mevsim == "Yaz" else 1.0)
            gunluk_saat = np.random.normal(0.5 * carpan, 0.1)
            mevsimsel_saatler[mevsim] += max(0, gunluk_saat)
            current += timedelta(days=1)


        #oyuncu seviyesi toplam oyun saatine bağlı olarak mantıklı şekilde hesaplanır.
        toplam_saat = sum(mevsimsel_saatler.values())
        seviye = max(1, min(100, int(np.sqrt(toplam_saat) * 2)))
        hesap_yasi = max(0.1, round(toplam_gun / 365.25, 1))

        #tüm bulunan veriler oyuncu verisi listesine eklenir ve dataframe yani tablo oluşturulur.
        oyuncu_verisi.append({
            "OyuncuID": 1000 + i,
            "Dogum_Tarihi": dogum_tarihi.strftime("%Y-%m-%d"),  # Yeni eklendi
            "Yas": yas,
            "Kayit_Tarihi": kayit_tarihi.strftime("%Y-%m-%d"),
            "Son_Giris": son_giris.strftime("%Y-%m-%d"),
            "Ilkbahar_Saati": int(mevsimsel_saatler["Ilkbahar"]),
            "Yaz_Saati": int(mevsimsel_saatler["Yaz"]),
            "Sonbahar_Saati": int(mevsimsel_saatler["Sonbahar"]),
            "Kis_Saati": int(mevsimsel_saatler["Kış"]),
            "Toplam_Saat": int(toplam_saat),
            "OyuncuSeviyesi": seviye,
            "Churn_Durumu": churn,
            "Hesap_Yasi": hesap_yasi
        })

    return pd.DataFrame(oyuncu_verisi)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    df = oyuncu_profili_uret(10000)

    df.to_csv(os.path.join(data_dir, "my_dataset.csv"), index=False)