import os
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Sayfa Konfigürasyonu
st.set_page_config(page_title="Player Churn Prediction Dashboard", layout="wide")

# Temel CSS Ayarları
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }
    .summary-card {
        background-color: var(--background-secondary-color, rgba(150, 150, 150, 0.15));
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        color: var(--text-color);
    }
    .metric-card {
        background-color: var(--background-secondary-color, rgba(150, 150, 150, 0.12));
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(150, 150, 150, 0.25);
    }
    .metric-title {
        font-size: 14px;
        opacity: 0.75;
        color: var(--text-color);
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: var(--text-color);
    }
    .feature-card {
        background-color: var(--background-secondary-color, rgba(150, 150, 150, 0.12));
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(150, 150, 150, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .feature-name {
        font-size: 15px;
        color: var(--text-color);
        font-weight: 500;
    }
    .feature-val {
        font-size: 18px;
        font-weight: bold;
        color: #2ecc71;
    }
    hr {
        display: none !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- YÜKLEME FONKSİYONLARI ---
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_data(data_path):
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


# --- BAŞLIK VE ÜST FİLTRE PANELİ ---
st.title("🎮 Oyuncu Churn Tahmin & Analiz Paneli")

col_filter, _ = st.columns([1, 2])
with col_filter:
    dataset_choice = st.selectbox(
        "📂 İncelenecek Veri Setini Seçin:", ["Kaggle Dataset", "My Dataset"]
    )

if dataset_choice == "Kaggle Dataset":
    data_file = (
        "data/kaggle_pipeline.csv"
        if os.path.exists("data/kaggle_pipeline.csv")
        else "data/kaggle_dataset.csv"
    )
    model_file = "saved_models/kaggle_rf_model.pkl"
else:
    data_file = (
        "data/my_pipeline.csv"
        if os.path.exists("data/my_pipeline.csv")
        else "data/my_dataset.csv"
    )
    model_file = "saved_models/my_rf_model.pkl"

df = load_data(data_file)
model = load_model(model_file)

if df is None:
    st.error(
        f"❌ Veri seti bulunamadı! Lütfen `{data_file}` dosyasının varlığını kontrol edin."
    )
    st.stop()

# Churn Sütununu Tespit Etme
actual_churn_col = (
    "Churn_Durumu"
    if "Churn_Durumu" in df.columns
    else ("Churn" if "Churn" in df.columns else None)
)

# Model Tahminlerini Üretme
ignore_cols = [
    "OyuncuID",
    "Dogum_Tarihi",
    "Kayit_Tarihi",
    "Son_Giris",
    "OyunaBaglilik",
    "Churn_Durumu",
    "Churn",
    "Churn_Tahmini",
]
X_cols = [col for col in df.columns if col not in ignore_cols]

if model is not None:
    try:
        df["Churn_Tahmini"] = model.predict(df[X_cols])
    except Exception:
        df["Churn_Tahmini"] = df[actual_churn_col] if actual_churn_col else 0
else:
    df["Churn_Tahmini"] = df[actual_churn_col] if actual_churn_col else 0

st.write("")

# --- 4'LÜ SEKME SİSTEMİ ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Genel Bakış"

tab_col1, tab_col2, tab_col3, tab_col4 = st.columns(4)

active1 = st.session_state.active_tab == "Genel Bakış"
active2 = st.session_state.active_tab == "Risk Altındaki Oyuncular"
active3 = st.session_state.active_tab == "Model Metrikleri"
active4 = st.session_state.active_tab == "Grafik"


with tab_col1:
    if st.button("📊 Genel Bakış", use_container_width=True, key="btn_t1"):
        st.session_state.active_tab = "Genel Bakış"
        st.rerun()

with tab_col2:

    if st.button("⚠️ Risk Altındaki Oyuncular", use_container_width=True, key="btn_t2"):
        st.session_state.active_tab = "Risk Altındaki Oyuncular"
        st.rerun()

with tab_col3:

    if st.button("📈 Model Metrikleri", use_container_width=True, key="btn_t3"):
        st.session_state.active_tab = "Model Metrikleri"
        st.rerun()

with tab_col4:

    if st.button("📊 Grafik", use_container_width=True, key="btn_t4"):
        st.session_state.active_tab = "Grafik"
        st.rerun()

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 1: GENEL BAKIŞ
# ---------------------------------------------------------
if st.session_state.active_tab == "Genel Bakış":
    st.subheader("📋 Genel Sistem & Oyuncu Durum Özeti")

    total_players = len(df)
    if actual_churn_col:
        churn_count = int((df[actual_churn_col] == 1).sum())
        retained_count = int((df[actual_churn_col] == 0).sum())
        churn_rate = (churn_count / total_players) * 100 if total_players > 0 else 0
        retained_rate = 100 - churn_rate
    else:
        churn_count = retained_count = churn_rate = retained_rate = 0

    col_metrics, col_info = st.columns([1, 1.2])

    with col_metrics:
        st.metric("Toplam Oyuncu Sayısı", f"{total_players:,}")
        st.metric(
            "Aktif Oyuncu Sayısı", f"{retained_count:,}", delta=f"%{retained_rate:.1f}"
        )
        st.metric(
            "Terk Eden Oyuncu Sayısı",
            f"{churn_count:,}",
            delta=f"%{churn_rate:.1f}",
            delta_color="inverse",
        )
        st.metric("Aktif Model Mimarisi", "Random Forest")

    with col_info:
        if dataset_choice == "Kaggle Dataset":
            st.markdown(
                """
            <div class="summary-card">
                <h4>📌 Kaggle Dataset Hakkında</h4>
                <ul><li><b>Kaynak:</b> <a href="https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset" target="_blank">Predict Online Gaming Behavior Dataset</a></li></ul>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="summary-card">
                <h4>📌 My Dataset Hakkında</h4>
                <p>Staj kapsamında kendi ürettiğim mevsimsel oyun saatleri içeren sentetik veri seti.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# TAB 2: RİSK ALTINDAKİ OYUNCULAR LİSTESİ
# ---------------------------------------------------------
elif st.session_state.active_tab == "Risk Altındaki Oyuncular":
    st.subheader("🎯 Risk Altındaki Oyuncular Listesi (Churn = 1)")

    if actual_churn_col:
        filtered_df = df[df[actual_churn_col] == 1].copy()

        search_id = st.text_input("🔍 Oyuncu ID ile Ara:", "")
        if search_id and "OyuncuID" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["OyuncuID"].astype(str).str.contains(search_id)
            ]

        cols_to_display = [
            c
            for c in filtered_df.columns
            if c not in ["Dogum_Tarihi", "Kayit_Tarihi", "Son_Giris", "Churn_Tahmini"]
        ]

        st.write(f"Risk grubunda toplam **{len(filtered_df)}** oyuncu listeleniyor.")
        st.dataframe(
            filtered_df[cols_to_display], use_container_width=True, hide_index=True
        )

        csv = filtered_df[cols_to_display].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Listeyi CSV Olarak İndir",
            data=csv,
            file_name=f"risk_altindaki_oyuncular_{dataset_choice.split()[0].lower()}.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "⚠️ Veri setinde Churn sütunu bulunamadığı için liste oluşturulamadı."
        )

# ---------------------------------------------------------
# TAB 3: MODEL METRİKLERİ
# ---------------------------------------------------------
elif st.session_state.active_tab == "Model Metrikleri":
    st.subheader("Random Forest Modeli Metrik Sonuçları")

    if dataset_choice == "Kaggle Dataset":
        m_precision, m_recall, m_f1, m_acc, m_cv = "%88", "%84", "%86", "%96", "%84.62"
    else:
        m_precision, m_recall, m_f1, m_acc, m_cv = "%85", "%88", "%87", "%88", "%86.70"

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{m_precision}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">{m_recall}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">F1-Score</div>
            <div class="metric-value">{m_f1}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{m_acc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with mc5:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Ortalama CV F1</div>
            <div class="metric-value">{m_cv}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Özellik Önem Düzeyleri")

    if dataset_choice == "Kaggle Dataset":
        features = [
            ("Toplam_Oyun_Suresi", "%52.3"),
            ("OturumSayisi", "%19.4"),
            ("Ortalama_Oturum_Suresi", "%14.0"),
            ("OyuncuSeviyesi", "%12.1"),
            ("Oturum_Basina_Harcama", "%1.6"),
            ("TotalSatınAlma", "%0.5"),
        ]
    else:
        features = [
            ("Gunluk_Oturum_Sikligi", "%31.6"),
            ("Ilerleme_Hizi", "%14.4"),
            ("Ilkbahar_Saati", "%10.9"),
            ("Yaz_Saati", "%9.4"),
            ("Toplam_Saat", "%7.7"),
            ("Kis_Saati", "%6.1"),
            ("Yas", "%6.0"),
        ]

    for fname, fval in features:
        st.markdown(
            f"""
        <div class="feature-card">
            <span class="feature-name">{fname}</span>
            <span class="feature-val">{fval}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# TAB 4: GRAFİK (OYUNCU DAVRANIŞ & TERK ANALİZLERİ)
# ---------------------------------------------------------
elif st.session_state.active_tab == "Grafik":
    st.subheader("📊 Terk Eden Oyuncu Davranış ve Dağılım Analizleri")
    st.write(
        f"**{dataset_choice}** verisinde oyunu terk eden oyuncuların detaylı analizleri:"
    )

    if actual_churn_col and dataset_choice == "Kaggle Dataset":
        churned_df = df[df[actual_churn_col] == 1].copy()

        # Grafik 1: Toplam Oyun Süresi Dağılımı
        st.markdown("#### ⏱️ Terk Edenlerin Toplam Oyun Süresi Dağılımı")
        if "Toplam_Oyun_Suresi" in churned_df.columns:
            bins = [0, 10, 20, 30, 40, 50, 100]
            labels = [
                "0-10 Saat",
                "10-20 Saat",
                "20-30 Saat",
                "30-40 Saat",
                "40-50 Saat",
                "50+ Saat",
            ]
            churned_df["Sure_Grubu"] = pd.cut(
                churned_df["Toplam_Oyun_Suresi"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            duration_counts = (
                churned_df["Sure_Grubu"]
                .value_counts()
                .reindex(labels, fill_value=0)
                .reset_index()
            )
            duration_counts.columns = ["Sure_Grubu", "Sayi"]

            chart1 = (
                alt.Chart(duration_counts)
                .mark_bar(color="#5294e2")
                .encode(
                    x=alt.X(
                        "Sure_Grubu:N",
                        sort=labels,
                        title="Süre Grubu",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("Sayi:Q", title="Oyuncu Sayısı"),
                )
                .properties(height=350)
            )
            st.altair_chart(chart1, use_container_width=True)
        else:
            st.info("Toplam oyun süresi sütunu bulunamadı.")

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        # Grafik 2: Seviye Aralıklarına Göre Dağılımı
        st.markdown("#### 🎮 Terk Edenlerin Seviye Aralıklarına Göre Dağılımı")
        level_col = "OyuncuSeviyesi" if "OyuncuSeviyesi" in churned_df.columns else None
        if level_col:
            max_level = (
                int(churned_df[level_col].max())
                if not churned_df[level_col].empty
                else 100
            )
            bins = list(range(0, max_level + 10, 10))
            labels = [f"{i}-{i + 10} Level" for i in bins[:-1]]

            churned_df["Level_Grubu"] = pd.cut(
                churned_df[level_col],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            level_group_counts = (
                churned_df["Level_Grubu"]
                .value_counts()
                .reindex(labels, fill_value=0)
                .reset_index()
            )
            level_group_counts.columns = ["Level_Grubu", "Sayi"]

            chart2 = (
                alt.Chart(level_group_counts)
                .mark_bar(color="#5294e2")
                .encode(
                    x=alt.X(
                        "Level_Grubu:N",
                        sort=labels,
                        title="Level Grubu",
                        axis=alt.Axis(labelAngle=-30),
                    ),
                    y=alt.Y("Sayi:Q", title="Oyuncu Sayısı"),
                )
                .properties(height=350)
            )
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.info("Oyuncu seviyesi sütunu bulunamadı.")

    elif actual_churn_col and dataset_choice == "My Dataset":
        churned_df = df[df[actual_churn_col] == 1].copy()

        # Grafik 3: Mevsimsel Oyun Saatleri
        st.markdown("#### ☀️ Terk Edenlerin Mevsimsel Oyun Saatleri")
        if "Yaz_Saati" in churned_df.columns and "Kis_Saati" in churned_df.columns:
            season_churn = pd.DataFrame(
                {
                    "Mevsim": ["İlkbahar", "Yaz", "Sonbahar", "Kış"],
                    "Ortalama_Saat": [
                        (
                            churned_df["Ilkbahar_Saati"].mean()
                            if "Ilkbahar_Saati" in churned_df.columns
                            else 0
                        ),
                        churned_df["Yaz_Saati"].mean(),
                        (
                            churned_df["Sonbahar_Saati"].mean()
                            if "Sonbahar_Saati" in churned_df.columns
                            else 0
                        ),
                        churned_df["Kis_Saati"].mean(),
                    ],
                }
            )

            chart3 = (
                alt.Chart(season_churn)
                .mark_bar(color="#5294e2")
                .encode(
                    x=alt.X(
                        "Mevsim:N",
                        sort=["İlkbahar", "Yaz", "Sonbahar", "Kış"],
                        title="Mevsim",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("Ortalama_Saat:Q", title="Ortalama Saat"),
                )
                .properties(height=350)
            )
            st.altair_chart(chart3, use_container_width=True)
        else:
            st.info("Mevsimsel saat sütunları bulunamadı.")

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        # Grafik 4: Yaş Dağılımı
        st.markdown("#### 👥 Terk Edenlerin Yaş Dağılımı")
        if "Yas" in churned_df.columns:
            age_counts = churned_df["Yas"].value_counts().sort_index().reset_index()
            age_counts.columns = ["Yas", "Sayi"]

            chart4 = (
                alt.Chart(age_counts)
                .mark_bar(color="#5294e2")
                .encode(
                    x=alt.X("Yas:O", title="Yaş", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Sayi:Q", title="Oyuncu Sayısı"),
                )
                .properties(height=350)
            )
            st.altair_chart(chart4, use_container_width=True)
        else:
            st.info("Yaş sütunu bulunamadı.")
    else:
        st.warning(
            "⚠️ Veri setinde Churn sütunu bulunamadığı için terk eden oyuncu analizleri yapılamıyor."
        )
