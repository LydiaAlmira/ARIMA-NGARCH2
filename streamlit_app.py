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
        "ARIMA-NGARCH (Prediksi)",
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
    - **ARIMA-NGARCH (Prediksi)**
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

    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu INPUT DATA \U0001F4C1")
        st.stop()

    if 'selected_vars' not in st.session_state or not st.session_state.selected_vars:
        st.warning("Silakan pilih variabel yang ingin diproses di INPUT DATA \U0001F4C1")
        st.stop()

    df = st.session_state.df.copy()
    currencies = st.session_state.selected_vars

    st.subheader("🔍 1. Cek Missing dan Duplicate")
    st.dataframe(df.isnull().sum())

    duplicates = df.duplicated()
    if duplicates.any():
        st.warning("Ditemukan baris duplikat:")
        st.dataframe(df[duplicates])
    else:
        st.success("✅ Tidak ada duplicated values.")

    st.subheader("\U0001F4CA 2. Statistik Deskriptif Harga")
    try:
        st.dataframe(df[currencies].describe())
    except KeyError as e:
        st.error(f"Error membaca kolom harga: {e}")
        st.stop()

    st.subheader("♻️ 3. Hitung Log-Return & Visualisasi")

    for currency in currencies:
        if currency not in df.columns:
            st.warning(f"Kolom {currency} tidak ditemukan.")
            continue

        if df[currency].max() > 100000:
            df[currency] = df[currency] / 1000

        df[f'{currency}_return'] = np.log(df[currency]).diff()

        st.markdown(f"##### Log-Return {currency}")
        st.line_chart(df[f'{currency}_return'].dropna())

    st.subheader("\U0001F4C8 4. Statistik Deskriptif Log-Return")

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
    import pandas as pd
    import numpy as np
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
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

    st.subheader("2️⃣ Pilih Orde ARIMA (p, d, q)")
    user_orders = {}
    for currency in currencies:
        st.markdown(f"#### {currency}")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input(f"p (AR) - {currency}", min_value=0, max_value=5, value=1, key=f"{currency}_p")
        with col2:
            d = st.number_input(f"d (I) - {currency}", min_value=0, max_value=2, value=0, key=f"{currency}_d")
        with col3:
            q = st.number_input(f"q (MA) - {currency}", min_value=0, max_value=5, value=1, key=f"{currency}_q")

        user_orders[currency] = (p, d, q)

    st.subheader("3️⃣ Estimasi Parameter ARIMA")
    model_fits = {}
    for currency, order in user_orders.items():
        try:
            model = ARIMA(train_data[currency], order=order).fit()
            model_fits[currency] = model
            st.markdown(f"### {currency} - ARIMA{order}")
            st.text(model.summary())
        except Exception as e:
            st.error(f"Gagal membangun model ARIMA untuk {currency}: {e}")

    st.session_state.arima_fits = model_fits
    st.session_state.arima_orders = user_orders

    st.subheader("4️⃣ Uji Asumsi Residual (Ljung-Box & Jarque-Bera)")
    ljungbox_results, jb_results = [], []

    for currency, model in model_fits.items():
        resid = model.resid.dropna()
        lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
        jb_stat, jb_pvalue = jarque_bera(resid)

        ljungbox_results.append({
            'Mata Uang': currency,
            'Model ARIMA': f"ARIMA{user_orders[currency]}",
            'Ljung-Box Stat': round(lb_test['lb_stat'].values[0], 4),
            'p-value': round(lb_test['lb_pvalue'].values[0], 4),
            'Keterangan': 'Tidak Autokorelasi' if lb_test['lb_pvalue'].values[0] > 0.05 else 'Ada Autokorelasi'
        })

        jb_results.append({
            'Mata Uang': currency,
            'Model ARIMA': f"ARIMA{user_orders[currency]}",
            'JB Stat': f"{jb_stat:.2f}",
            'p-value': f"{jb_pvalue:.4f}" if jb_pvalue >= 0.0001 else '0.0000',
            'Keterangan': 'Normal' if jb_pvalue > 0.05 else 'Tidak Normal'
        })

    st.markdown("#### Hasil Uji Ljung-Box")
    st.dataframe(pd.DataFrame(ljungbox_results))
    st.markdown("#### Hasil Uji Jarque-Bera")
    st.dataframe(pd.DataFrame(jb_results))
    
    # === SIMPAN MODEL SIGNIFIKAN UNTUK GARCH ===
    model_fits_signifikan = {}
    for result in ljungbox_results:
        currency = result['Mata Uang']
        if result['p-value'] > 0.05:  # Tidak ada autokorelasi → layak lanjut ke GARCH
            model_fits_signifikan[currency] = model_fits[currency]

    st.session_state.model_fits_signifikan = model_fits_signifikan

    st.subheader("5️⃣ Prediksi Data Test & Evaluasi Akurasi")
    result_price_all = {}
    mape_scores = {}

    for currency, model in model_fits.items():
        forecast_return = model.forecast(steps=len(test_data[currency]))
        last_train_index = train_data[currency].index[-1]
        last_price = df.loc[last_train_index, currency]
        forecast_price = last_price * np.exp(np.cumsum(forecast_return))
        actual_price = df.loc[test_data[currency].index, currency]

        result_df = pd.DataFrame({
            'Actual': actual_price.values,
            'Forecast': forecast_price.values
        }, index=test_data[currency].index)

        result_price_all[currency] = result_df

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            nonzero = y_true != 0
            return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

        mape = mean_absolute_percentage_error(result_df['Actual'], result_df['Forecast'])
        mape_scores[currency] = round(mape, 2)

        st.markdown(f"### {currency}")
        st.dataframe(result_df)
        st.line_chart(result_df)

    st.markdown("#### MAPE (%) Harga")
    st.dataframe(pd.DataFrame.from_dict(mape_scores, orient='index', columns=['MAPE (%)']))

    st.subheader("6️⃣ Prediksi 30 Hari ke Depan")
    future_forecasts = {}
    forecast_df = pd.DataFrame()
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30, freq='D')

    for currency, model in model_fits.items():
        forecast_return = model.forecast(steps=30)
        last_train_index = train_data[currency].index[-1]
        last_price = df.loc[last_train_index, currency]
        forecast_price = last_price * np.exp(np.cumsum(forecast_return))
        forecast_df[f"Prediksi Harga {currency}"] = forecast_price.values

    forecast_df.insert(0, 'Tanggal', future_dates)
    st.dataframe(forecast_df)
    st.line_chart(forecast_df.set_index('Tanggal'))

    st.subheader("7️⃣ Residual Diagnostics (ACF/PACF & Uji ARCH-LM)")
    arch_lm_results = []

    model_config = st.session_state.arima_orders  # Ambil konfigurasi ARIMA

    for currency, order in model_config.items():
        st.markdown(f"#### Residual Analysis - {currency} (ARIMA{order})")

        series = train_data[currency]
        model = ARIMA(series, order=order).fit()
        residuals = model.resid.dropna()

        # Plot ACF dan PACF
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(residuals, lags=20, ax=ax[0])
        ax[0].set_title(f"ACF Residual - {currency}")

        plot_pacf(residuals, lags=20, ax=ax[1])
        ax[1].set_title(f"PACF Residual - {currency}")

        fig.suptitle(f"ACF & PACF Residual - ARIMA{order} (Return {currency})", fontsize=14)
        st.pyplot(fig)

        # Uji ARCH LM
        arch_stat, arch_pvalue, _, _ = het_arch(residuals)
        arch_lm_results.append({
            'Mata Uang': currency,
            'ARCH Stat': f"{arch_stat:.2f}",
            'p-value': f"{arch_pvalue:.4f}" if arch_pvalue >= 0.0001 else '<0.0001',
            'Keterangan': (
                'Tidak Ada Efek ARCH' if arch_pvalue > 0.05 else 'Ada Efek ARCH'
            )
        })

    arch_lm_df = pd.DataFrame(arch_lm_results)
    st.markdown("### Hasil Uji ARCH LM pada Residual Model ARIMA (Return)")
    st.dataframe(arch_lm_df)

    st.success("Residual analysis selesai. Siap lanjut ke pemodelan GARCH!")

elif menu == "GARCH (Model)":
    st.header("📉 GARCH Modeling & Prediction")
    st.write("Analisis volatilitas dengan model GARCH, evaluasi residual, dan uji non-linearitas.")

    import pandas as pd
    import numpy as np
    from arch import arch_model
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # Ambil residual dari ARIMA
    if 'arima_fits' not in st.session_state:
        st.warning("Silakan jalankan ARIMA terlebih dahulu.")
        st.stop()

    model_fits = st.session_state.arima_fits
    currencies = list(model_fits.keys())

    st.subheader("1️⃣ ACF & PACF dari Residual Kuadrat ARIMA")
    for currency in currencies:
        residuals = model_fits[currency].resid.dropna()
        squared_resid = residuals ** 2

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(squared_resid, ax=ax[0], lags=20)
        ax[0].set_title(f"ACF Squared Residual - {currency}")

        plot_pacf(squared_resid, ax=ax[1], lags=20)
        ax[1].set_title(f"PACF Squared Residual - {currency}")

        st.pyplot(fig)

    st.subheader("2️⃣ Pilih Orde GARCH untuk Setiap Mata Uang")
    garch_orders = {}
    for currency in currencies:
        p = st.selectbox(f"Pilih p (ARCH) untuk {currency}", [1, 2], key=f"p_{currency}")
        q = st.selectbox(f"Pilih q (GARCH) untuk {currency}", [1, 2], key=f"q_{currency}")
        garch_orders[currency] = (p, q)

    st.subheader("3️⃣ Estimasi Parameter GARCH")
    garch_fits = {}
    for currency, (p, q) in garch_orders.items():
        resid = model_fits[currency].resid.dropna()
        model = arch_model(resid, vol='GARCH', p=p, q=q, mean='Zero')
        result = model.fit(disp='off')
        garch_fits[f"{currency}_GARCH({p},{q})"] = result

        st.markdown(f"**{currency} - GARCH({p},{q})**")
        st.text(result.summary())

    st.session_state.garch_fits = garch_fits

    st.subheader("4️⃣ Uji Asumsi Residual GARCH (Ljung-Box & ARCH LM)")
    ljungbox_rows = []
    archlm_rows = []

    for name, result in garch_fits.items():
        resid = result.resid.dropna()

        # Ljung-Box
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_stat = float(lb['lb_stat'].iloc[0])
        lb_p = float(lb['lb_pvalue'].iloc[0])

        # ARCH LM
        arch_stat, arch_p, _, _ = het_arch(resid)
        arch_p = float(arch_p)

        ljungbox_rows.append({
            'Model': name,
            'LB Stat': round(lb_stat, 4),
            'p-value': round(lb_p, 4),
            'Keterangan': 'Tidak ada autokorelasi' if lb_p > 0.05 else 'Ada autokorelasi'
        })

        archlm_rows.append({
            'Model': name,
            'ARCH-LM Stat': round(arch_stat, 4),
            'p-value': round(arch_p, 4),
            'Keterangan': 'Tidak ada heteroskedastisitas' if arch_p > 0.05 else 'Ada heteroskedastisitas'
        })

    st.markdown("📊 Hasil Uji Ljung-Box:")
    st.dataframe(pd.DataFrame(ljungbox_rows))

    st.markdown("📊 Hasil Uji ARCH-LM:")
    st.dataframe(pd.DataFrame(archlm_rows))

    st.subheader("5️⃣ Uji Non-Linearitas (Terasvirta Neural Test)")

    def neural_test_squared_resid(residuals, lags=1):
        resid_sq = residuals ** 2
        lagged_resid = pd.concat([resid_sq.shift(i) for i in range(1, lags + 1)], axis=1)
        lagged_resid.columns = [f"Lag_{i}" for i in range(1, lags + 1)]

        data = pd.concat([resid_sq, lagged_resid], axis=1).dropna()
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]

        X['Lag1_sq'] = X.iloc[:, 0] ** 2
        X['Lag1_cube'] = X.iloc[:, 0] ** 3

        X_linear = sm.add_constant(X.iloc[:, :lags])
        model_linear = sm.OLS(y, X_linear).fit()

        X_full = sm.add_constant(X)
        model_nonlinear = sm.OLS(y, X_full).fit()

        f_stat, p_value, _ = model_nonlinear.compare_f_test(model_linear)
        return {
            'F-statistic': round(f_stat, 4),
            'p-value': round(p_value, 4),
            'Kesimpulan': 'Ada non-linearitas' if p_value < 0.05 else 'Tidak ada non-linearitas'
        }

    nonlinear_results = []
    for name, result in garch_fits.items():
        resid = result.std_resid.dropna()
        res = neural_test_squared_resid(resid)
        res['Model'] = name
        nonlinear_results.append(res)

    st.markdown("📊 Hasil Uji Non-Linearitas (Terasvirta Neural Test):")
    st.dataframe(pd.DataFrame(nonlinear_results)[['Model', 'F-statistic', 'p-value', 'Kesimpulan']])

    st.success("✅ Analisis GARCH selesai. Siap lanjut ke NGARCH 🚀")
    

elif menu == "ARIMA-NGARCH (Prediksi)":
    st.title("🔀 ARIMA-NGARCH: Prediksi Harga dan Volatilitas")

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    df = st.session_state.df_processed

    model_fits_signifikan = st.session_state.get("model_fits_signifikan", {})
    train_data = st.session_state.train_data
    test_data = st.session_state.test_data
    df = st.session_state.df_processed

    for currency in ['IDR', 'MYR', 'SGD']:
        st.subheader(f"🔹 {currency}")

        if currency not in model_fits_signifikan:
            st.warning(f"Model ARIMA untuk {currency} tidak tersedia.")
            continue

        if currency == 'IDR':
            omega, alpha, beta, gamma = 8.7206e-07, 0.1000, 0.8800, 0.0102
            model_type = 'NGARCH(1,1)'
        elif currency == 'MYR':
            omega, alpha1, alpha2, beta, gamma = 7.9481e-07, 0.0906, 0.0905, 0.7300, 0.0382
            model_type = 'NGARCH(2,1)'
        elif currency == 'SGD':
            omega, alpha, beta, gamma = 1.5322e-07, 0.0500, 0.9300, -0.0146
            model_type = 'NGARCH(1,1)'

        if currency not in train_data or currency not in test_data:
            st.warning(f"Data return untuk {currency} tidak ditemukan.")
            continue

        returns_all = pd.concat([train_data[currency], test_data[currency]]).reset_index(drop=True)
        T = len(returns_all)
        h = np.zeros(T)

        if currency == 'MYR':
            h[0:2] = np.var(train_data[currency])
            for t in range(2, T):
                eps1 = returns_all.iloc[t - 1]
                eps2 = returns_all.iloc[t - 2]
                term1 = alpha1 * (eps1 - gamma * np.sqrt(h[t - 1]))**2
                term2 = alpha2 * (eps2 - gamma * np.sqrt(h[t - 2]))**2
                h[t] = omega + term1 + term2 + beta * h[t - 1]
        else:
            h[0] = np.var(train_data[currency])
            for t in range(1, T):
                eps_prev = returns_all.iloc[t - 1]
                h[t] = omega + alpha * (eps_prev - gamma * np.sqrt(h[t - 1]))**2 + beta * h[t - 1]

        forecasted_vol = np.sqrt(h[-30:])

        forecast_return = model_fits_signifikan[currency].forecast(steps=30).reset_index(drop=True)

        last_train_index = train_data[currency].index[-1]
        last_price = df.loc[last_train_index, currency]

        forecast_price = last_price * np.exp(np.cumsum(forecast_return))
        upper_band = forecast_price * np.exp(forecasted_vol)
        lower_band = forecast_price * np.exp(-forecasted_vol)

        test_index = test_data[currency].index
        actual_price = df.loc[test_index, currency].values[:30]

        result_df = pd.DataFrame({
            'Hari': range(1, 31),
            'Actual': actual_price,
            'Forecast': forecast_price.values,
            'Upper_Band': upper_band.values,
            'Lower_Band': lower_band.values
        })

        st.markdown(f"📘 **Model:** ARIMA + {model_type}")
        st.dataframe(result_df, use_container_width=True)
        st.line_chart(result_df.set_index('Hari')[['Actual', 'Forecast']])

        realized_var = test_data[currency][:30]**2
        forecasted_var = forecasted_vol**2
        rmse = np.sqrt(mean_squared_error(realized_var, forecasted_var))
        mae = mean_absolute_error(realized_var, forecasted_var)
        mape = np.mean(np.abs((realized_var - forecasted_var) / realized_var)) * 100

        st.write("📊 **Evaluasi Prediksi Volatilitas (Squared Return vs Variance):**")
        st.write(f"**RMSE:** {rmse:.6f}")
        st.write(f"**MAE:** {mae:.6f}")
        st.write(f"**MAPE:** {mape:.2f}%")
        st.markdown("---")
