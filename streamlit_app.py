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

            # Simpan ke session
            st.session_state.df = df.copy()

            st.subheader("🔍 Data Setelah Diproses")
            st.dataframe(df)

            # Pilih variabel numerik
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            if not numeric_cols:
                st.error("❌ Tidak ditemukan kolom numerik setelah pembersihan.")
            else:
                selected_vars = st.multiselect("Pilih variabel mata uang yang ingin diproses:", numeric_cols)

                if selected_vars:
                    st.session_state.selected_vars = selected_vars
                    st.success(f"Variabel terpilih: {', '.join(selected_vars)}")
                else:
                    st.warning("Pilih minimal satu variabel numerik untuk diproses.")

                # Visualisasi pertama (opsional)
                selected_chart = st.selectbox("Pilih salah satu untuk ditampilkan grafik:", selected_vars)
                st.line_chart(df[selected_chart])

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera
    from datetime import timedelta

    st.header("📈 ARIMA Modeling & Forecasting")
    st.write("Model ARIMA untuk memodelkan log-return nilai tukar dan prediksi harga.")

    if 'train_data' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")
        st.stop()

    train_data = st.session_state.train_data
    test_data = st.session_state.test_data
    df = st.session_state.df_processed
    currencies = list(train_data.keys())

    st.subheader("1️⃣ Identifikasi Model (ACF & PACF)")
    for currency in currencies:
        st.markdown(f"#### {currency}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(train_data[currency], ax=ax[0], lags=20)
        ax[0].set_title(f"ACF {currency} Return")
        plot_pacf(train_data[currency], ax=ax[1], lags=20)
        ax[1].set_title(f"PACF {currency} Return")
        st.pyplot(fig)

    st.subheader("2️⃣ Pemilihan Model Berdasarkan AIC")
    candidate_orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]
    best_models = []

    for currency in currencies:
        best_aic = float('inf')
        best_order = None
        for order in candidate_orders:
            try:
                model = ARIMA(train_data[currency], order=order).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
            except:
                continue
        best_models.append({'Mata Uang': currency, 'Order': f'ARIMA{best_order}', 'AIC': round(best_aic, 2)})

    df_best_arima = pd.DataFrame(best_models)
    st.dataframe(df_best_arima)

    st.subheader("3️⃣ Uji Asumsi Residual (Ljung-Box & Jarque-Bera)")
    model_config = {
        'IDR': (2, 0, 1),
        'MYR': (1, 0, 1),
        'SGD': (1, 0, 0)
    }
    ljungbox_results, jb_results = [], []
    model_fits = {}

    for currency, order in model_config.items():
        model = ARIMA(train_data[currency], order=order).fit()
        model_fits[currency] = model

        resid = model.resid.dropna()
        lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
        jb_stat, jb_pvalue = jarque_bera(resid)

        ljungbox_results.append({
            'Mata Uang': currency,
            'Model': f'ARIMA{order}',
            'Ljung-Box Stat': lb_test['lb_stat'].values[0],
            'p-value': lb_test['lb_pvalue'].values[0],
            'Autokorelasi': 'Tidak' if lb_test['lb_pvalue'].values[0] > 0.05 else 'Ada'
        })

        jb_results.append({
            'Mata Uang': currency,
            'Model': f'ARIMA{order}',
            'JB Stat': f"{jb_stat:.2f}",
            'p-value': f"{jb_pvalue:.4f}" if jb_pvalue >= 0.0001 else '0.0000',
            'Normalitas': 'Normal' if jb_pvalue > 0.05 else 'Tidak Normal'
        })

    st.markdown("#### Hasil Uji Ljung-Box")
    st.dataframe(pd.DataFrame(ljungbox_results))
    st.markdown("#### Hasil Uji Jarque-Bera")
    st.dataframe(pd.DataFrame(jb_results))

    st.subheader("4️⃣ Prediksi Test Data (30 Hari Terakhir)")
    result_price_all = {}

    for currency in currencies:
        forecast_return = model_fits[currency].forecast(steps=len(test_data[currency]))
        last_price = df.loc[train_data[currency].index[-1], currency]
        forecast_price = last_price * np.exp(np.cumsum(forecast_return))
        actual_price = df.loc[test_data[currency].index, currency]

        result_df = pd.DataFrame({
            'Actual': actual_price.values,
            'Forecast': forecast_price.values
        }, index=test_data[currency].index)

        result_price_all[currency] = result_df

        st.markdown(f"#### {currency}: Harga Aktual vs Prediksi")
        st.line_chart(result_df)

    st.subheader("📊 Evaluasi Akurasi (MAPE)")
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        nonzero = y_true != 0
        return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    mape_scores = {currency: mean_absolute_percentage_error(df_['Actual'], df_['Forecast'])
                   for currency, df_ in result_price_all.items()}
    st.dataframe(pd.DataFrame.from_dict(mape_scores, orient='index', columns=['MAPE (%)']))

    st.subheader("5️⃣ Prediksi Harga 30 Hari ke Depan")
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30, freq='D')
    future_preds = {}

    for currency in currencies:
        forecast_return = model_fits[currency].forecast(steps=30)
        last_price = df.loc[train_data[currency].index[-1], currency]
        future_price = last_price * np.exp(np.cumsum(forecast_return))
        future_preds[currency] = future_price

    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi Harga IDR': future_preds['IDR'].values,
        'Prediksi Harga MYR': future_preds['MYR'].values,
        'Prediksi Harga SGD': future_preds['SGD'].values,
    })

    st.dataframe(forecast_df)
    st.line_chart(forecast_df.set_index('Tanggal'))


elif menu == "GARCH (Model & Prediksi)":
    st.header("GARCH Model & Prediksi")
    st.write("... kode GARCH ...")

elif menu == "NGARCH (Model & Prediksi)":
    st.header("NGARCH Model & Prediksi")
    st.write("... kode NGARCH ...")

