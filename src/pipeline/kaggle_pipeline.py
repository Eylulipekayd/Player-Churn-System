import pandas as pd
import numpy as np


def load_data(file_path):
    """ Veriyi diskten okur"""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """ Eksik/aykırı değerleri temizler"""

    # Tabloda boş bırakılmış satırlar varsa onları kalıcı olarak siliyoruz.
    df = df.dropna()

    # Aykırı değer temizliği (negatif değer varsa sıfırlar.)
    for sutun in df.columns:

        # ID, Churn ve metin tabanlı bağlılık sütununu temizliğin dışında tutuyoruz
        if sutun in ['OyuncuID', 'OyunaBaglilik']:
            continue

        df[sutun] = np.where(df[sutun] < 0, 0, df[sutun])
    return df


def add_features(df):
    """ Özellik mühendisliği (Yeni sütunlar ekleme)"""
    # Toplam oyun süresini hesaplar.
    # Kaggle verisinde toplam süre yoktu, bu iki metriği çarparak toplam süreyi buluyoruz.
    df['Toplam_Oyun_Suresi'] = df['OturumSayisi'] * df['Ortalama_Oturum_Suresi']

    # Oturum başına harcamayı hesaplar.
    # Oyuncunun her oyuna girdiğinde ortalama ne kadar para harcadığını gösterir.
    # Oturum sayısı 0 olan satırlarda hata almamak için np.where kullanıyoruz.
    df['Oturum_Basina_Harcama'] = np.where(df['OturumSayisi'] > 0, df['TotalSatınAlma'] / df['OturumSayisi'], 0)

    if 'OyunaBaglilik' in df.columns:
        df['Churn'] = np.where(df['OyunaBaglilik'] == 'Low', 1, 0)
    else:
        df['Churn'] = 0
    return df


def save_processed_data(df, output_path):
    """ İşlenmiş son tabloyu kaydeder"""
    df.to_csv(output_path, index=False)

def run_pipeline():
        print("Pipeline başlatıldı...")

        # Oku
        read_df = load_data("data/kaggle_dataset.csv")

        #  Temizle
        cleaned_df = clean_data(read_df)

        #  Özellikleri üret
        featured_df = add_features(cleaned_df)

        #  Kaydet
        save_processed_data(featured_df, "data/kaggle_pipeline.csv")

        print("Pipeline başarıyla tamamlandı! Veri kaydedildi.")

if __name__ == "__main__":
    run_pipeline()
