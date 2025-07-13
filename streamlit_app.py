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
    st.header("📁 Input Data Nilai Tukar (Format Penelitian)")
    uploaded_file = st.file_uploader("Upload file CSV (delimiter = ;)", type="csv")

    if uploaded_file is not None:
        import pandas as pd

        try:
            # Baca file dengan delimiter ;
            df = pd.read_csv(uploaded_file, delimiter=';')

            # Format tanggal
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M').dt.strftime('%d/%m/%Y')

            # Bersihkan nilai dari titik dan koma
            for col in df.columns[1:]:
                df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)

            # Set index dan urutkan
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # Konversi ke numerik
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

            # Pastikan index datetime
            df.index = pd.to_datetime(df.index, dayfirst=True)
            df = df.sort_index()

            # Tampilkan hasil
            st.subheader("🔍 Data Setelah Diproses")
            st.dataframe(df)

            # Pilih kolom untuk divisualisasikan
            st.subheader("📈 Visualisasi Time Series")
            selected_col = st.selectbox("Pilih kolom mata uang:", df.columns.tolist())
            st.line_chart(df[selected_col])

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca data: {e}")
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

