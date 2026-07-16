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

    # 6. ROC Eğrisi Çizimi
    y_probs_base = baseline_model.predict_proba(X_test)[:, 1]
    y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

    fpr_base, tpr_base, _ = roc_curve(y_test, y_probs_base)
    auc_base = auc(fpr_base, tpr_base)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    dark_bg = '#1e1e1e'
    text_color = '#e0e0e0'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    # Modellerin Eğrilerini Çizdirme
    ax.plot(fpr_base, tpr_base, color='#00b0ff', linewidth=2.5,
            label=f'Logistic Regression (AUC = {auc_base:.4f})')
    ax.plot(fpr_rf, tpr_rf, color='#00e676', linewidth=2.5,
            label=f'Random Forest (AUC = {auc_rf:.4f})')


    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.5, label='Rastgele Tahmin (AUC = 0.5000)')


    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.xaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)
    ax.yaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_color('#333333')

    ax.tick_params(colors=text_color, which='both', labelsize=11)
    ax.set_xlabel('Yalancı Alarm Oranı (False Positive Rate)', color=text_color, fontsize=11, labelpad=10)
    ax.set_ylabel('Doğru Yakalama Oranı (True Positive Rate)', color=text_color, fontsize=11, labelpad=10)
    ax.set_title('Kaggle Veri Seti: ROC Eğrisi Karşılaştırması', color=text_color, fontsize=13, fontweight='bold',
                 pad=15)

    legend = ax.legend(loc='lower right', facecolor='#252525', edgecolor='#333333', fontsize=10)
    plt.setp(legend.get_texts(), color=text_color)

    plt.tight_layout()

    roc_plot_path = os.path.join(reports_dir, "kaggle_roc_auc.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"✓OC Eğrisi grafiği kaydedildi: {roc_plot_path}")

    # 7. Özellik önemlerinin çıkarılması
    importances = rf_model.feature_importances_
    feature_names = X.columns

    # Özellik önemlerini DataFrame'e döküp sıralıyoruz
    feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)
    bars = ax.barh(feat_importances.index, feat_importances.values, color='#00e676', edgecolor='#333333', height=0.5)

    ax.xaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7)
    ax.yaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_color('#333333')

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

    feat_plot_path = os.path.join(reports_dir, "kaggle_feature_importances.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"✓ Özellik önem dereceleri grafiği kaydedildi: {feat_plot_path}")


if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_path = os.path.join(base_dir, "data", "kaggle_pipeline.csv")
    reports_dir = os.path.join(base_dir, "reports")

    print("=== KAGGLE TRAİN PİPELİNE BAŞLATILDI ===")

    df = load_processed_data(data_path)
    X, y = prepare_features(df)
    train_and_evaluate(X, y, reports_dir)

    print("\n=== TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI ===")