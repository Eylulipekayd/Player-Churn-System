import os
import joblib
import pandas as pd


def load_my_model(model_path):
    """Kaydedilmiş kendi Random Forest modelini yükler."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hata: Model dosyası bulunamadı: {model_path}")

    model = joblib.load(model_path)
    print(f"✓ Model başarıyla yüklendi: {os.path.basename(model_path)}")
    return model


def predict_my_churn(model, input_data):
    """
    Kendi veri seti formatındaki oyuncu verisi için Churn tahmini hesaplar.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Girdi verisi bir dictionary veya DataFrame olmalıdır.")

    # Model ile tahmin yapma
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]  # Churn (1) olma olasılığı (%)

    # Sonuçları ekleyip döndürme
    df['Churn_Tahmini'] = predictions
    df['Churn_Olasiligi_%'] = (probabilities * 100).round(2)

    return df


if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 1. Model Dosya Yolu
    model_path = os.path.join(base_dir, "saved_models", "my_rf_model.pkl")

    print("=== MY DATASET CHURN TAHMİN SİSTEMİ (PREDICTION) BAŞLATILDI ===")

    # 2. Modeli Yükle
    model = load_my_model(model_path)
    data_path = os.path.join(base_dir, "data", "my_pipeline.csv")

    if os.path.exists(data_path):
        sample_df = pd.read_csv(data_path).head(5)  # İlk 5 oyuncuyu alalım

        # Eğitime girmeyen sütunları çıkarıyoruz
        sutunlar_to_drop = ['OyuncuID', 'Dogum_Tarihi', 'Kayit_Tarihi', 'Son_Giris', 'Churn_Durumu']
        X_sample = sample_df.drop(columns=[col for col in sutunlar_to_drop if col in sample_df.columns])

        # 4. Tahmin
        results = predict_my_churn(model, X_sample)

        print("\n=== ÖRNEK 5 OYUNCU İÇİN TAHMİN SONUÇLARI ===")
        print(results[['Churn_Tahmini', 'Churn_Olasiligi_%']])
    else:
        print(f"Test verisi bulunamadı: {data_path}")