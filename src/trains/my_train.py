import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt


def load_processed_data(file_path):
    """Temizlenmiş pipeline veri yükler."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Hata: {file_path} bulunamadı!\nDosya yolunu kontrol edin: {os.path.abspath(file_path)}")

    df = pd.read_csv(file_path)
    print("Veri başarıyla yüklendi. Toplam satır sayısı:", len(df))
    return df


def prepare_features(df):
    sutunlar_to_drop_my = ['OyuncuID', 'Dogum_Tarihi', 'Kayit_Tarihi', 'Son_Giris', 'Churn_Durumu']

    X_my = df.drop(columns=[col for col in sutunlar_to_drop_my if col in df.columns])
    Y_my = df['Churn_Durumu']

    print("\nModelin kullanacağı girdiler sütun sayısı:", X_my.shape[1])
    print("Sütunların Listesi:", X_my.columns.tolist())
    return X_my, Y_my


def train_and_evaluate(X_my, Y_my, reports_dir):
    """Modelleri eğitir, çapraz doğrulama yapar ve grafikleri kaydeder."""
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Train-Test
    X_train_my, X_test_my, Y_train_my, Y_test_my = train_test_split(
        X_my, Y_my,
        test_size=0.20,
        random_state=42,
        stratify=Y_my
    )
    print(f"\nEğitim Seti Boyutu (X_train_my): {X_train_my.shape}")
    print(f"Test Seti Boyutu (X_test_my): {X_test_my.shape}")

    # 2. Baseline Model (Logistic Regression)
    baseline_my = LogisticRegression(max_iter=3000, random_state=42)
    baseline_my.fit(X_train_my, Y_train_my)
    Y_pred_base_my = baseline_my.predict(X_test_my)

    print("\n=== BASELINE MODEL (LOGISTIC REGRESSION) PERFORMANSI ===\n")
    print(classification_report(Y_test_my, Y_pred_base_my))

    # 3. Ağaç Tabanlı Model (Random Forest)
    rf_my = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_my.fit(X_train_my, Y_train_my)
    Y_pred_rf_my = rf_my.predict(X_test_my)

    print("\n=== AĞAÇ TABANLI MODEL (RANDOM FOREST) PERFORMANSI ===\n")
    print(classification_report(Y_test_my, Y_pred_rf_my))

    # 4. Çapraz Doğrulama (Cross-Validation)
    cv_scores_my = cross_val_score(rf_my, X_train_my, Y_train_my, cv=5, scoring='f1')
    print("\n=== ÇAPRAZ DOĞRULAMA SONUÇLARI ===")
    clean_scores_my = [round(float(score), 4) for score in cv_scores_my]
    print("Her katın F1 skoru:", clean_scores_my)
    print(f"Ortalama F1 Skoru  : {cv_scores_my.mean():.4f}")
    print(f"Standart Sapma     : {cv_scores_my.std():.4f}")

    # 5. Karmaşıklık Matrislerini Çizdirme ve Kaydetme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_base_my = confusion_matrix(Y_test_my, Y_pred_base_my)
    disp_base_my = ConfusionMatrixDisplay(confusion_matrix=cm_base_my, display_labels=["Churn (0)", "Churn(1)"])
    disp_base_my.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Logistic Regression (Baseline)')

    cm_rf_my = confusion_matrix(Y_test_my, Y_pred_rf_my)
    disp_rf_my = ConfusionMatrixDisplay(confusion_matrix=cm_rf_my, display_labels=["Churn(0)", "Churn(1)"])
    disp_rf_my.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('Random Forest')

    plt.suptitle('Karmaşıklık Matrisi Karşılaştırması', fontsize=14, y=1.05)
    plt.tight_layout()

    cm_plot_path = os.path.join(reports_dir, "my_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"\n✓ Karmaşıklık matrisi grafiği kaydedildi: {cm_plot_path}")

    # 6. ROC Eğrisi Çizimi
    Y_probs_base_my = baseline_my.predict_proba(X_test_my)[:, 1]
    Y_probs_rf_my = rf_my.predict_proba(X_test_my)[:, 1]

    fpr_base_my, tpr_base_my, _ = roc_curve(Y_test_my, Y_probs_base_my)
    roc_auc_base_my = auc(fpr_base_my, tpr_base_my)

    fpr_rf_my, tpr_rf_my, _ = roc_curve(Y_test_my, Y_probs_rf_my)
    roc_auc_rf_my = auc(fpr_rf_my, tpr_rf_my)

    # Grafik Alanı Tanımlama
    dark_bg = '#1e1e1e'
    text_color = '#e0e0e0'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    # Modellerin Eğrilerini Çizdirme
    ax.plot(fpr_base_my, tpr_base_my, color='#00b0ff', linewidth=2.5,
            label=f'Logistic Regression (AUC = {roc_auc_base_my:.4f})')
    ax.plot(fpr_rf_my, tpr_rf_my, color='#00e676', linewidth=2.5,
            label=f'Random Forest (AUC = {roc_auc_rf_my:.4f})')

    # Rastgele Tahmin Çizgisi (Kılavuz Çizgi)
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.5, label='Rastgele Tahmin (AUC = 0.5000)')

    # Grafik Sınırları ve Izgara Ayarları
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.xaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)
    ax.yaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_color('#333333')

    ax.tick_params(colors=text_color, which='both', labelsize=11)
    ax.set_xlabel('Yalancı Alarm Oranı (False Positive Rate)', color=text_color, fontsize=11, labelpad=10)
    ax.set_ylabel('Doğru Yakalama Oranı (True Positive Rate)', color=text_color, fontsize=11, labelpad=10)
    ax.set_title('ROC Eğrisi Karşılaştırması', color=text_color, fontsize=13, fontweight='bold',
                 pad=15)

    # Lejant (Açıklama Kutusu) Ayarları
    legend = ax.legend(loc='lower right', facecolor='#252525', edgecolor='#333333', fontsize=10)
    plt.setp(legend.get_texts(), color=text_color)

    plt.tight_layout()

    roc_plot_path = os.path.join(reports_dir, "my_roc_auc.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"✓ROC Eğrisi grafiği kaydedildi: {roc_plot_path}")

    # 7. Özellik önemleri

    importances = rf_my.feature_importances_
    feature_names = X_my.columns

    # Verileri bir DataFrame'de toplayıp büyükten küçüğe sıralıyoruz
    feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)
    bars = ax.barh(feat_importances.index, feat_importances.values, color='#00e676', edgecolor='#333333', height=0.6)

    # Izgara çizgileri ve kenarlık tasarımları
    ax.xaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)
    ax.yaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    # Barların üzerine yüzde değerlerini yazıyoruz
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'%{width * 100:.1f}',
                va='center', ha='left',
                color=text_color, fontweight='bold', fontsize=10)

    ax.tick_params(colors=text_color, which='both', labelsize=11)
    ax.set_xlabel('Önem Derecesi (0 - 1 Arası)', color=text_color, fontsize=11, labelpad=10)
    ax.set_title('Modelin Karar Verirken En Çok Önem Verdiği Özellikler', color=text_color,
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    # Grafiği reports klasörüne kaydediyoruz
    feat_plot_path = os.path.join(reports_dir, "my_feature_importances.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"✓ Özellik önem dereceleri grafiği kaydedildi: {feat_plot_path}")


if __name__ == "__main__":
    # trains klasörünün iki seviye yukarısındaki ana klasörü buluyoruz
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_path = os.path.join(base_dir, "data", "my_pipeline.csv")
    reports_dir = os.path.join(base_dir, "reports", "Hafta_3", "train_results")

    print("=== MY TRAİN PİPELİNE BAŞLATILDI ===")

    df_my = load_processed_data(data_path)
    X_my, Y_my = prepare_features(df_my)
    train_and_evaluate(X_my, Y_my, reports_dir)

    print("\n=== TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI ===")