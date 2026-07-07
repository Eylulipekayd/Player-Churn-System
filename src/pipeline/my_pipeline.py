import pandas as pd
import numpy as np


def load_data(file_path):
    """ Veriyi diskten okur"""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """ Eksik/aykırı değerleri temizler veya doldurur"""

    # Tabloda boş bırakılmış satırlar varsa onları kalıcı olarak siliyoruz.
    df = df.dropna()

    # Aykırı değer temizliği (negatif değer varsa sıfırlar.)
    for sutun in df.columns:

        # Tarih içeren, ID olan veya Churn durumu belirten sütunları temizliğin dışında tutuyoruz
        if sutun in ['OyuncuID', 'Dogum_Tarihi', 'Kayit_Tarihi', 'Son_Giris', 'Churn_Durumu']:
            continue

        df[sutun] = np.where(df[sutun] < 0, 0, df[sutun])

    return df


def add_features(df):
    """ Özellik mühendisliği (Yeni sütunlar ekleme)"""

    return df


def save_processed_data(df, output_path):
    """ İşlenmiş son tabloyu kaydeder"""
    df.to_csv(output_path, index=False)


def run_pipeline():
    print("Pipeline başlatıldı...")

    #  Oku
    read_df = load_data("data/my_dataset.csv")

    #  Temizle
    cleaned_df = clean_data(read_df)

    #  Özellikleri üret
    featured_df = add_features(cleaned_df)

    #  Kaydet
    save_processed_data(featured_df, "data/my_pipeline.csv")

    print("Pipeline başarıyla tamamlandı! Veri kaydedildi.")


if __name__ == "__main__":
    run_pipeline()