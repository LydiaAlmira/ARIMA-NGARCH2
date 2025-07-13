import streamlit as st

# === CONFIGURASI UMUM ===
st.set_page_config(
    page_title="ARIMA-NGARCH Prediksi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === SIDEBAR NAVIGASI ===
menu = st.sidebar.radio(
    "📋 MENU NAVIGASI:",
    [
        "HOME 🏠",
        "INPUT DATA 📁",
        "DATA PREPROCESSING 🧹",
        "STASIONERITAS DATA 📉",
        "DATA SPLITTING ✂️",
        "ARIMA (Model & Prediksi)",
        "GARCH (Model & Prediksi)",
        "NGARCH (Model & Prediksi)",
        "INTERPRETASI & SARAN 💡"
    ]
)

# === TAMPILAN UTAMA BERDASARKAN MENU ===
st.markdown("<h1 style='text-align: center; color: white; background-color:#1f77b4; padding: 15px; border-radius: 10px;'>"
            "Prediksi Data Time Series Univariat<br>Menggunakan Model ARIMA-NGARCH ✏️</h1>",
            unsafe_allow_html=True)

if menu == "HOME 🏠":
    st.info("Sistem ini dirancang untuk melakukan prediksi nilai tukar menggunakan model ARIMA dan mengukur volatilitasnya dengan model NGARCH. 📊 📈")
    
    st.markdown("### Panduan Penggunaan Sistem 🔎")
    st.markdown("""
    - **HOME** 🏠: Penjelasan sistem.
    - **INPUT DATA** 📁: Unggah data nilai tukar (`.csv`).
    - **DATA PREPROCESSING** 🧹: Transformasi data dan return.
    - **STASIONERITAS DATA** 📉: Uji ADF dan ACF/PACF.
    - **DATA SPLITTING** ✂️: Bagi data train/test.
    - **ARIMA (Model & Prediksi)**: Bangun model ARIMA:
        - Uji signifikan residual
        - Prediksi nilai tukar
        - Evaluasi performa (RMSE, MAE, MAPE)
    - **GARCH (Model & Prediksi)**: GARCH pada residual ARIMA.
    - **NGARCH (Model & Prediksi)**: Volatilitas lanjutan.
    - **INTERPRETASI & SARAN** 💡: Kesimpulan hasil.
    """)

elif menu == "INPUT DATA 📁":
    st.header("📁 Input Data Nilai Tukar")
    
    uploaded_file = st.file_uploader("Upload file CSV berisi data time series nilai tukar:", type="csv")

    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Data Preview")
        st.dataframe(df)

        # Coba cari kolom numerik untuk diplot
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if numeric_cols:
            st.subheader("📈 Visualisasi Data (Line Chart)")
            selected_col = st.selectbox("Pilih kolom yang ingin divisualisasikan:", numeric_cols)
            st.line_chart(df[selected_col])
        else:
            st.warning("Tidak ditemukan kolom numerik untuk divisualisasikan.")
    else:
        st.info("Silakan unggah file CSV terlebih dahulu.")


elif menu == "DATA PREPROCESSING 🧹":
    st.header("🧹 Preprocessing Data")
    st.write("... kode preprocessing di sini ...")

elif menu == "STASIONERITAS DATA 📉":
    st.header("📉 Uji Stasioneritas")
    st.write("... kode ADF, ACF, PACF ...")

elif menu == "DATA SPLITTING ✂️":
    st.header("✂️ Pembagian Data")
    st.write("... kode split train/test ...")

elif menu == "ARIMA (Model & Prediksi)":
    st.header("ARIMA Model & Prediksi")
    st.write("... kode ARIMA, prediksi & evaluasi ...")

elif menu == "GARCH (Model & Prediksi)":
    st.header("GARCH Model & Prediksi")
    st.write("... kode GARCH ...")

elif menu == "NGARCH (Model & Prediksi)":
    st.header("NGARCH Model & Prediksi")
    st.write("... kode NGARCH ...")

