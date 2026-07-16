import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt


def load_processed_data(file_path):
    """Kaggle temizlenmiş pipeline verisini yükler."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Hata: {file_path} bulunamadı!\nDosya yolunu kontrol edin: {os.path.abspath(file_path)}")

    df = pd.read_csv(file_path)
    print("Veri başarıyla yüklendi. Toplam satır sayısı:", len(df))
    return df


def prepare_features(df):
    sutunlar_to_drop = ['Churn', 'OyuncuID', 'OyunaBaglilik']

    X = df.drop(columns=[col for col in sutunlar_to_drop if col in df.columns])
    y = df['Churn']

    print("\nModelin kullanacağı girdiler sütun sayısı:", X.shape[1])
    print("Sütunların Listesi:", X.columns.tolist())
    return X, y


def train_and_evaluate(X, y, reports_dir):
    """Modelleri eğitir, çapraz doğrulama yapar ve grafikleri kaydeder."""
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Train-Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain Set Boyutu: {X_train.shape}")
    print(f"Test Set Boyutu: {X_test.shape}")

    # 2. Baseline Model (Logistic Regression)
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    print("\n=== BASELINE MODEL (LOGISTIC REGRESSION) PERFORMANSI ===")
    print(classification_report(y_test, y_pred_baseline))

    # 3. Ağaç Tabanlı Model (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("\n=== AĞAÇ TABANLI MODEL (RANDOM FOREST) PERFORMANSI ===")
    print(classification_report(y_test, y_pred_rf))

    # 4.Çapraz Doğrulama
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
    print("\n=== ÇAPRAZ DOĞRULAMA SONUÇLARI ===")

    clean_scores = [round(float(score), 4) for score in cv_scores]
    print("Her bir katın F1 Skoru:", clean_scores)
    print(f"Ortalama F1 Skoru: {cv_scores.mean():.4f}")
    print(f"F1 Skorlarının Standart Sapması: {cv_scores.std():.4f}")

    # 5. Karmaşıklık Matrislerini Çizdirme ve Kaydetme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_base = confusion_matrix(y_test, y_pred_baseline)
    disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=["Churn(0)", "Churn(1)"])
    disp_base.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Logistic Regression (Baseline)')

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Churn(0)", "Churn(1)"])
    disp_rf.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('Random Forest')

    plt.suptitle('Karmaşıklık Matrisi Karşılaştırması', fontsize=14, y=1.05)
    plt.tight_layout()

    # Karmaşıklık matrisini kaydet
    cm_plot_path = os.path.join(reports_dir, "kaggle_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"\n✓ Karmaşıklık matrisi grafiği kaydedildi: {cm_plot_path}")

    # 6. Dinamik ROC-AUC Hesaplama ve Görselleştirme
    y_probs_base = baseline_model.predict_proba(X_test)[:, 1]
    y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

    fpr_base, tpr_base, _ = roc_curve(y_test, y_probs_base)
    auc_base = auc(fpr_base, tpr_base)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    model_names = ['Logistic Regression\n(Baseline)', 'Random Forest\n(Ağaç Tabanlı)']
    auc_scores = [auc_base, auc_rf]

    dark_bg = '#1e1e1e'
    text_color = '#e0e0e0'
    bar_colors = ['#00b0ff', '#00e676']

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    bars = ax.barh(model_names, auc_scores, color=bar_colors, height=0.5, edgecolor='#333333', linewidth=1.2)
    ax.xaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)
    ax.yaxis.grid(False)

    for bar in bars:
        width = bar.get_width()
        ax.text(width - 0.12,
                bar.get_y() + bar.get_height() / 2,
                f'AUC: {width:.4f}',
                va='center', ha='left',
                color='#ffffff', fontweight='bold', fontsize=11)

    ax.set_xlim([0.80, 1.01])

    for spine in ax.spines.values():
        spine.set_color('#333333')

    ax.tick_params(colors=text_color, which='both', labelsize=11)
    ax.set_xlabel('ROC-AUC Skoru', color=text_color, fontsize=11, labelpad=10)
    ax.set_title('Modellerin ROC-AUC Performans Karşılaştırması', color=text_color, fontsize=13, fontweight='bold',
                 pad=15)

    plt.tight_layout()

    # ROC-AUC grafiğini kaydet
    roc_plot_path = os.path.join(reports_dir, "kaggle_roc_auc.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"✓ ROC-AUC karşılaştırma grafiği kaydedildi: {roc_plot_path}")


if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_path = os.path.join(base_dir, "data", "kaggle_pipeline.csv")
    reports_dir = os.path.join(base_dir, "reports")

    print("=== KAGGLE TRAIN PIPELINE BAŞLATILDI ===")

    df = load_processed_data(data_path)
    X, y = prepare_features(df)
    train_and_evaluate(X, y, reports_dir)

    print("\n=== TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI ===")