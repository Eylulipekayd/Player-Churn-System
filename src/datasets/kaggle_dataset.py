import os
import pandas as pd


def find_col(columns, keywords):
    """
    Sütun isimleri arasında belirtilen anahtar kelimeleri içeren ilk sütunu bulur.
    Büyük/küçük harf duyarlılığını ortadan kaldırır.
    """
    for col in columns:
        if all(k in col.lower() for k in keywords):
            return col
    return None


def player_data(input_path, output_path):

    # Csv dosyasını okur ve sütun isimlerinde boşluk varsa siler
    df_data = pd.read_csv(input_path)
    df_data.columns = df_data.columns.str.strip()

    # find_col fonksiyonuna keyword gider ve içinde geçen sütun bulunur
    id_col = find_col(df_data.columns, ["id"])
    session_col = find_col(df_data.columns, ["session"])
    engagement_col = find_col(df_data.columns, ["engagement"])

    # Engagement" ve Level kelimeleri bazen yan yana gelip EngagementLevel çakışabiliyor.
    # Bu yüzden Level bilgisini ararken Engagement sütununu devre dışı bırakıyor
    level_search_cols = [c for c in df_data.columns if c != engagement_col]
    level_col = find_col(level_search_cols, ["level"])

    duration_col = find_col(df_data.columns, ["duration"]) or find_col(
        df_data.columns, ["avg"]
    )
    purchase_col = find_col(df_data.columns, ["purchase"]) or find_col(
        df_data.columns, ["buy"]
    )

    # Veri setindeki isimleri, projede kullanacağım sütun isimleriyle eşleştiriyor.
    columns_mapping = {}
    if id_col:
        columns_mapping[id_col] = "OyuncuID"
    if session_col:
        columns_mapping[session_col] = "OturumSayisi"
    if level_col:
        columns_mapping[level_col] = "OyuncuSeviyesi"
    if duration_col:
        columns_mapping[duration_col] = "Ortalama_Oturum_Suresi"
    if purchase_col:
        columns_mapping[purchase_col] = "TotalSatınAlma"
    if engagement_col:
        columns_mapping[engagement_col] = "OyunaBaglilik"

    df_filtered = df_data.rename(columns=columns_mapping)

    # Veri setinin sütunları gerekli sütunların içindeki isimler gibi  olsun.
    gerekli_sutunlar = [
        "OyuncuID",
        "OturumSayisi",
        "OyuncuSeviyesi",
        "TotalSatınAlma",
        "Ortalama_Oturum_Suresi",
        "OyunaBaglilik",
    ]

    # Filtrelenmiş veri setinden 10.000 satırı seçer
    df_filtered = (
        df_filtered[gerekli_sutunlar]
        .sample(n=min(10000, len(df_filtered)), random_state=42)
        .reset_index(drop=True)
    )
    # Filtrelenmiş veri setini kaydeder
    df_filtered.to_csv(output_path, index=False)

    # Veri setinin olduğu csv okuyacağı input ve filtrelenmiş halini kaydedeceğimiz output belirtir.


if __name__ == "__main__":
    # Doğru yazımı budur: dirname ile dosyanın klasörünü alıp, join ile iki üst klasöre çıkıyoruz
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    input_file = os.path.join(base_dir, "data", "gaming_data.csv")
    output_file = os.path.join(base_dir, "data", "kaggle_dataset.csv")

    player_data(input_file, output_file)
