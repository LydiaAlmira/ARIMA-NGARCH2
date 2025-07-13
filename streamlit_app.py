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
        "ARIMA (Model & Prediksi)",
        "GARCH (Model)",
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
    - **ARIMA (Model & Prediksi)**: Bangun model ARIMA:
        - Uji signifikan residual
        - Prediksi nilai tukar
        - Evaluasi performa (RMSE, MAE, MAPE)
    - **GARCH (Model)**: GARCH pada residual ARIMA.
    - **NGARCH (Model & Prediksi)**: Volatilitas lanjutan.
    """)

elif menu == "INPUT DATA 📁":
    st.header("📁 Input Data Nilai Tukar (Format Penelitian)")
    uploaded_file = st.file_uploader("Upload file CSV (delimiter = ;)", type="csv")

    if uploaded_file is not None:
        import pandas as pd

        try:
            # Baca file dengan delimiter ;
            df = pd.read_csv(uploaded_file, delimiter=';')
            st.session_state.df = df.copy()
       
            # Pilih variabel yang ingin dianalisis
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            selected_vars = st.multiselect("Pilih variabel mata uang yang ingin diproses:", numeric_cols)

            if selected_vars:
                st.session_state.selected_vars = selected_vars
                st.success(f"Variabel terpilih: {', '.join(selected_vars)}")
            else:
                st.warning("Pilih minimal satu variabel numerik untuk diproses.")

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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller

    st.header("🧹 Data Cleaning, Log-Return & ADF Test")
    st.write("Lakukan pembersihan, transformasi return log, dan uji stasioneritas.")

    # Cek apakah file sudah diinput
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu INPUT DATA 📁")
        st.stop()

    if 'selected_vars' not in st.session_state or not st.session_state.selected_vars:
        st.warning("Silakan pilih variabel yang ingin diproses di INPUT DATA 📁")
        st.stop()

    df = st.session_state.df.copy()
    currencies = st.session_state.selected_vars

    st.subheader("🔍 1. Cek Missing dan Duplicate")
    st.write("Jumlah missing values:")
    st.dataframe(df.isnull().sum())

    duplicates = df.duplicated()
    if duplicates.any():
        st.warning("Ditemukan baris duplikat:")
        st.dataframe(df[duplicates])
    else:
        st.success("✅ Tidak ada duplicated values.")

    st.subheader("📊 2. Statistik Deskriptif Harga")
    try:
        st.dataframe(df[currencies].describe())
    except KeyError as e:
        st.error(f"Error membaca kolom harga: {e}")
        st.stop()

    st.subheader("🔁 3. Hitung Log-Return & Visualisasi")

    for currency in currencies:
        if currency not in df.columns:
            st.warning(f"Kolom {currency} tidak ditemukan.")
            continue

        # Penyesuaian skala
        if df[currency].max() > 100000:
            df[currency] = df[currency] / 1000

        # Hitung log-return
        df[f'{currency}_return'] = np.log(df[currency]).diff()

        # Visualisasi
        st.markdown(f"##### Log-Return {currency}")
        st.line_chart(df[f'{currency}_return'].dropna())

    st.subheader("📈 4. Statistik Deskriptif Log-Return")

    for currency in currencies:
        col_name = f"{currency}_return"
        if col_name in df.columns:
            stats = df[col_name].dropna().describe()
            st.write(f"**{currency}**")
            st.dataframe(stats.to_frame())
        else:
            st.warning(f"Log-return untuk {currency} belum dihitung.")

    st.subheader("✂️ 5. Split Data: Train & Test (30 baris terakhir sebagai test)")
    train_data = {}
    test_data = {}

    for currency in currencies:
        col_name = f"{currency}_return"
        if col_name not in df.columns:
            continue

        return_series = df[col_name].dropna()
        if len(return_series) < 31:
            st.warning(f"Data log-return {currency} terlalu pendek untuk split.")
            continue

        train = return_series.iloc[:-30]
        test = return_series.iloc[-30:]

        train_data[currency] = train
        test_data[currency] = test

        st.write(f"**{currency}** - Train: {train.shape[0]}, Test: {test.shape[0]}")

    st.session_state.train_data = train_data
    st.session_state.test_data = test_data
    st.session_state.df_processed = df

    st.subheader("🧪 6. Uji Stasioneritas ADF (log-return train)")

    for currency in currencies:
        if currency not in train_data:
            st.warning(f"Data train untuk {currency} tidak tersedia.")
            continue

        st.markdown(f"**{currency}**")
        result = adfuller(train_data[currency])
        adf_stat = result[0]
        p_value = result[1]

        st.write(f"ADF Statistic : {adf_stat:.6f}")
        st.write(f"p-value       : {p_value:.6f}")

        if p_value < 0.05:
            st.success("✅ Data stasioner (tolak H0)")
        else:
            st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")


elif menu == "ARIMA (Model & Prediksi)":
    st.header("ARIMA Model & Prediksi")
    st.write("... kode ARIMA, prediksi & evaluasi ...")

elif menu == "GARCH (Model & Prediksi)":
    st.header("GARCH Model & Prediksi")
    st.write("... kode GARCH ...")

elif menu == "NGARCH (Model & Prediksi)":
    st.header("NGARCH Model & Prediksi")
    st.write("... kode NGARCH ...")

