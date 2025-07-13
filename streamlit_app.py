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

            # Pilih variabel numerik (hanya satu)
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            if not numeric_cols:
                st.error("❌ Tidak ditemukan kolom numerik setelah pembersihan.")
            else:
                selected_currency = st.selectbox("Pilih salah satu variabel mata uang untuk diproses:", numeric_cols)

                if selected_currency:
                    st.session_state.selected_currency = selected_currency
                    st.success(f"Variabel terpilih: {selected_currency}")

                    # Visualisasi
                    st.line_chart(df[selected_currency])
                else:
                    st.warning("Harap pilih satu variabel mata uang.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca data: {e}")
    else:
        st.info("Silakan unggah file CSV terlebih dahulu.")


elif menu == "DATA PREPROCESSING \U0001F9F9":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller

    st.header("\U0001F9F9 Data Cleaning, Log-Return & ADF Test")
    st.write("Lakukan pembersihan, transformasi return log, dan uji stasioneritas.")

    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu INPUT DATA \U0001F4C1")
        st.stop()

    if 'selected_currency' not in st.session_state:
        st.warning("Silakan pilih satu variabel terlebih dahulu di menu INPUT DATA \U0001F4C1")
        st.stop()

    df = st.session_state.df.copy()
    currency = st.session_state.selected_currency

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
        st.dataframe(df[currency].describe())
    except KeyError as e:
        st.error(f"Error membaca kolom harga: {e}")
        st.stop()

    st.subheader("♻️ 3. Hitung Log-Return & Visualisasi")

    if currency not in df.columns:
        st.warning(f"Kolom {currency} tidak ditemukan.")
    else:
        if df[currency].max() > 100000:
            df[currency] = df[currency] / 1000

        df[f'{currency}_return'] = np.log(df[currency]).diff()

        st.markdown(f"##### Log-Return {currency}")
        st.line_chart(df[f'{currency}_return'].dropna())

    st.subheader("\U0001F4C8 4. Statistik Deskriptif Log-Return")

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

    if col_name in df.columns:
        return_series = df[col_name].dropna()
        if len(return_series) < 31:
            st.warning(f"Data log-return {currency} terlalu pendek untuk split.")
        else:
            train = return_series.iloc[:-30]
            test = return_series.iloc[-30:]
            train_data[currency] = train
            test_data[currency] = test
            st.write(f"**{currency}** - Train: {train.shape[0]}, Test: {test.shape[0]}")

    st.session_state.train_data = train_data
    st.session_state.test_data = test_data
    st.session_state.df_processed = df

    st.subheader("\U0001F9EA 6. Uji Stasioneritas ADF (log-return train)")

    if currency in train_data:
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
    else:
        st.warning(f"Data train untuk {currency} tidak tersedia.")


elif menu == "ARIMA (Model & Prediksi)":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from scipy.stats import jarque_bera
    from datetime import timedelta

    st.header("\U0001F4C8 ARIMA Modeling & Forecasting")
    st.write("Model ARIMA untuk memodelkan log-return nilai tukar dan prediksi harga.")

    if 'train_data' not in st.session_state or 'selected_currency' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data dan pilih mata uang terlebih dahulu.")
        st.stop()

    train_data = st.session_state.train_data
    test_data = st.session_state.test_data
    df = st.session_state.df_processed
    currency = st.session_state.selected_currency

    st.subheader("1️⃣ Identifikasi Model (ACF & PACF)")
    st.markdown(f"#### {currency}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train_data[currency], ax=ax[0], lags=20)
    ax[0].set_title(f"ACF {currency} Return")
    plot_pacf(train_data[currency], ax=ax[1], lags=20)
    ax[1].set_title(f"PACF {currency} Return")
    st.pyplot(fig)

    st.subheader("2️⃣ Pilih Orde ARIMA (p, d, q)")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input(f"p (AR) - {currency}", min_value=0, max_value=5, value=1, key=f"{currency}_p")
    with col2:
        d = st.number_input(f"d (I) - {currency}", min_value=0, max_value=2, value=0, key=f"{currency}_d")
    with col3:
        q = st.number_input(f"q (MA) - {currency}", min_value=0, max_value=5, value=1, key=f"{currency}_q")

    order = (p, d, q)

    st.subheader("3️⃣ Estimasi Parameter ARIMA")
    try:
        model = ARIMA(train_data[currency], order=order).fit()
        st.session_state.arima_fit = model
        st.session_state.arima_order = order

        st.markdown(f"### {currency} - ARIMA{order}")
        st.text(model.summary())
    except Exception as e:
        st.error(f"Gagal membangun model ARIMA untuk {currency}: {e}")
        st.stop()

    st.subheader("4️⃣ Uji Asumsi Residual (Ljung-Box & Jarque-Bera)")
    resid = model.resid.dropna()
    lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
    jb_stat, jb_pvalue = jarque_bera(resid)

    ljungbox_result = pd.DataFrame([{
        'Model': f"ARIMA{order}",
        'LB Stat': round(lb_test['lb_stat'].values[0], 4),
        'p-value': round(lb_test['lb_pvalue'].values[0], 4),
        'Keterangan': 'Tidak Autokorelasi' if lb_test['lb_pvalue'].values[0] > 0.05 else 'Ada Autokorelasi'
    }])

    jb_result = pd.DataFrame([{
        'Model': f"ARIMA{order}",
        'JB Stat': f"{jb_stat:.2f}",
        'p-value': f"{jb_pvalue:.4f}" if jb_pvalue >= 0.0001 else '0.0000',
        'Keterangan': 'Normal' if jb_pvalue > 0.05 else 'Tidak Normal'
    }])

    st.markdown("#### Hasil Uji Ljung-Box")
    st.dataframe(ljungbox_result)
    st.markdown("#### Hasil Uji Jarque-Bera")
    st.dataframe(jb_result)

    st.subheader("5️⃣ Prediksi Data Test & Evaluasi Akurasi")
    forecast_return = model.forecast(steps=len(test_data[currency]))
    last_train_index = train_data[currency].index[-1]
    last_price = df.loc[last_train_index, currency]
    forecast_price = last_price * np.exp(np.cumsum(forecast_return))
    actual_price = df.loc[test_data[currency].index, currency]

    result_df = pd.DataFrame({
        'Actual': actual_price.values,
        'Forecast': forecast_price.values
    }, index=test_data[currency].index)

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        nonzero = y_true != 0
        return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    mape = mean_absolute_percentage_error(result_df['Actual'], result_df['Forecast'])

    st.markdown(f"### {currency}")
    st.dataframe(result_df)
    st.line_chart(result_df)

    st.markdown("#### MAPE (%) Harga")
    st.dataframe(pd.DataFrame({currency: [round(mape, 2)]}, index=['MAPE (%)']).T)

    st.subheader("6️⃣ Prediksi 30 Hari ke Depan")
    forecast_30 = model.forecast(steps=30)
    last_price = df.loc[train_data[currency].index[-1], currency]
    forecast_price_30 = last_price * np.exp(np.cumsum(forecast_30))
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30, freq='D')
    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        f"Prediksi Harga {currency}": forecast_price_30
    })

    st.dataframe(forecast_df)
    st.line_chart(forecast_df.set_index('Tanggal'))

    st.subheader("7️⃣ Residual Diagnostics (ACF/PACF & Uji ARCH-LM)")
    residuals = model.resid.dropna()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, lags=20, ax=ax[0])
    ax[0].set_title(f"ACF Residual - {currency}")
    plot_pacf(residuals, lags=20, ax=ax[1])
    ax[1].set_title(f"PACF Residual - {currency}")
    st.pyplot(fig)

    arch_stat, arch_pvalue, _, _ = het_arch(residuals)
    arch_result = pd.DataFrame([{
        'ARCH Stat': f"{arch_stat:.2f}",
        'p-value': f"{arch_pvalue:.4f}" if arch_pvalue >= 0.0001 else '<0.0001',
        'Keterangan': 'Tidak Ada Efek ARCH' if arch_pvalue > 0.05 else 'Ada Efek ARCH'
    }])

    st.markdown("### Hasil Uji ARCH LM pada Residual Model ARIMA")
    st.dataframe(arch_result)

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

    if "selected_vars" not in st.session_state or not st.session_state.selected_vars:
        st.warning("Silakan pilih variabel terlebih dahulu di menu INPUT DATA 📁")
        st.stop()

    selected_currency = st.session_state.selected_vars[0]

    # Ambil model ARIMA yang sudah dibentuk sebelumnya
    if 'arima_fits' not in st.session_state:
        st.warning("Silakan jalankan ARIMA terlebih dahulu.")
        st.stop()

    if selected_currency not in st.session_state.arima_fits:
        st.warning(f"Model ARIMA untuk {selected_currency} tidak tersedia.")
        st.stop()

    residuals = st.session_state.arima_fits[selected_currency].resid.dropna()
    squared_resid = residuals ** 2

    st.subheader("1️⃣ ACF & PACF dari Residual Kuadrat ARIMA")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(squared_resid, ax=ax[0], lags=20)
    ax[0].set_title(f"ACF Squared Residual - {selected_currency}")
    plot_pacf(squared_resid, ax=ax[1], lags=20)
    ax[1].set_title(f"PACF Squared Residual - {selected_currency}")
    st.pyplot(fig)

    st.subheader("2️⃣ Pilih Orde GARCH")
    p = st.selectbox(f"Pilih p (ARCH)", [1, 2], key="garch_p")
    q = st.selectbox(f"Pilih q (GARCH)", [1, 2], key="garch_q")

    st.subheader("3️⃣ Estimasi Parameter GARCH")
    model = arch_model(residuals, vol='GARCH', p=p, q=q, mean='Zero')
    result = model.fit(disp='off')
    st.session_state.garch_fit = result  # Simpan hasil untuk digunakan di NGARCH

    st.markdown(f"**{selected_currency} - GARCH({p},{q})**")
    st.text(result.summary())

    st.subheader("4️⃣ Uji Asumsi Residual GARCH (Ljung-Box & ARCH LM)")
    resid = result.resid.dropna()

    # Ljung-Box
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_stat = float(lb['lb_stat'].iloc[0])
    lb_p = float(lb['lb_pvalue'].iloc[0])
    lb_ket = 'Tidak ada autokorelasi' if lb_p > 0.05 else 'Ada autokorelasi'

    # ARCH LM
    arch_stat, arch_p, _, _ = het_arch(resid)
    arch_ket = 'Tidak ada heteroskedastisitas' if arch_p > 0.05 else 'Ada heteroskedastisitas'

    st.write("📊 **Hasil Uji Ljung-Box**")
    st.write(f"Statistik: {lb_stat:.4f}, p-value: {lb_p:.4f} → {lb_ket}")

    st.write("📊 **Hasil Uji ARCH-LM**")
    st.write(f"Statistik: {arch_stat:.4f}, p-value: {arch_p:.4f} → {arch_ket}")

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
        return f_stat, p_value

    f_stat, p_value = neural_test_squared_resid(result.std_resid.dropna())

    kesimpulan = 'Ada non-linearitas' if p_value < 0.05 else 'Tidak ada non-linearitas'
    st.write("📊 **Hasil Uji Terasvirta**")
    st.write(f"F-stat: {f_stat:.4f}, p-value: {p_value:.4f} → {kesimpulan}")

    st.success("✅ Analisis GARCH selesai. Siap lanjut ke NGARCH 🚀")

    
elif menu == "NGARCH (Model & Prediksi)":
    st.header("🔁 NGARCH(1,1) Modeling & Forecast")
    st.write("Estimasi volatilitas dengan model NGARCH(1,1) menggunakan Maximum Likelihood Estimation.")

    import pandas as pd
    import numpy as np
    from scipy.optimize import minimize
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from scipy.stats import probplot
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Ambil data dari ARIMA/GARCH
    if 'train_data' not in st.session_state or 'test_data' not in st.session_state:
        st.warning("Pastikan data telah diolah di ARIMA atau GARCH terlebih dahulu.")
        st.stop()

    train_data = st.session_state.train_data
    test_data = st.session_state.test_data

    # Gunakan return SGD gabungan (train + test)
    returns_all = pd.concat([train_data['SGD'], test_data['SGD']]).reset_index(drop=True)
    returns_all = returns_all.dropna()

    # Hitung gamma manual
    mean_r = np.mean(returns_all)
    std_r = np.std(returns_all)
    gamma_manual = mean_r / std_r

    # Log-likelihood NGARCH
    def ngarch_loglik(params, returns):
        omega, alpha, beta, gamma = params
        T = len(returns)
        h = np.zeros(T)
        h[0] = np.var(returns)

        for t in range(1, T):
            h[t] = omega + alpha * (returns[t-1] - gamma * np.sqrt(h[t-1]))**2 + beta * h[t-1]

        h = np.maximum(h, 1e-8)
        ll = -0.5 * (np.log(2*np.pi) + np.log(h) + (returns**2)/h)
        return -np.sum(ll)

    # Estimasi parameter
    initial_params = [1e-7, 0.05, 0.9, gamma_manual]
    bounds = [(1e-10, None), (1e-6, 1), (1e-6, 1), (-1, 1)]
    result = minimize(ngarch_loglik, initial_params, args=(returns_all,), method='L-BFGS-B', bounds=bounds)
    omega, alpha, beta, gamma = result.x

    st.subheader("1️⃣ Estimasi Parameter NGARCH(1,1)")
    st.text(f"omega : {omega:.6e}\nalpha : {alpha:.4f}\nbeta  : {beta:.4f}\ngamma : {gamma:.4f}\n\nLog-Likelihood: {-result.fun:.4f}")

    # Hitung h(t)
    T = len(returns_all)
    h = np.zeros(T)
    h[0] = np.var(returns_all)

    for t in range(1, T):
        eps_prev = returns_all.iloc[t - 1]
        h[t] = omega + alpha * (eps_prev - gamma * np.sqrt(h[t - 1]))**2 + beta * h[t - 1]

    h = np.maximum(h, 1e-8)
    std_resid_ngarch = returns_all / np.sqrt(h)

    st.subheader("2️⃣ Uji Asumsi Residual NGARCH")
    resid = std_resid_ngarch
    resid_sq = resid ** 2

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(resid, kde=True, bins=30, ax=ax[0], color='lightblue')
    ax[0].set_title("Histogram + KDE Residual Standar")
    probplot(resid, dist="norm", plot=ax[1])
    ax[1].set_title("QQ-Plot Residual Standar")
    st.pyplot(fig)

    st.write("\n**Ljung-Box Test (lag=10):**")
    lb_test = acorr_ljungbox(resid_sq, lags=[10], return_df=True)
    st.dataframe(lb_test)

    arch_stat, arch_p, _, _ = het_arch(resid)
    arch_ket = 'Tidak ada efek ARCH residual' if arch_p > 0.05 else 'Ada efek ARCH residual'
    st.markdown(f"**ARCH-LM Test:** p-value = {arch_p:.4f} → {arch_ket}")

    st.subheader("3️⃣ Prediksi Volatilitas Data Test (30 Hari)")
    T_full = len(returns_all)
    forecasted_vol_test = np.sqrt(h[-30:])
    st.line_chart(pd.Series(forecasted_vol_test, name="Volatilitas", index=range(1, 31)))

    st.subheader("4️⃣ Prediksi Volatilitas ke Depan (30 Hari")
    n_forecast = 30
    h_future = np.zeros(n_forecast)
    h_future[0] = h[-1]
    for t in range(1, n_forecast):
        h_future[t] = omega + alpha * (-gamma * np.sqrt(h_future[t-1]))**2 + beta * h_future[t-1]

    forecasted_vol_future = np.sqrt(h_future)
    st.line_chart(pd.Series(forecasted_vol_future, name="Prediksi Vol ke Depan"))

    st.subheader("5️⃣ Evaluasi Akurasi Prediksi")
    realized_var = test_data['SGD'] ** 2
    forecasted_var = forecasted_vol_test ** 2
    rmse = np.sqrt(mean_squared_error(realized_var, forecasted_var))
    mae = mean_absolute_error(realized_var, forecasted_var)
    st.write(f"**RMSE**: {rmse:.8f}")
    st.write(f"**MAE**: {mae:.8f}")


elif menu == "ARIMA-NGARCH (Prediksi)":
    st.title("🔀 ARIMA-NGARCH: Prediksi Harga dan Volatilitas")

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    if "selected_vars" not in st.session_state or not st.session_state.selected_vars:
        st.warning("❗Silakan pilih variabel terlebih dahulu di menu INPUT DATA 📁")
        st.stop()

    selected_currency = st.session_state.selected_vars[0]  # hanya satu variabel dipilih
    df = st.session_state.df_processed
    train_data = st.session_state.train_data
    test_data = st.session_state.test_data
    model_fits_signifikan = st.session_state.get("model_fits_signifikan", {})

    st.subheader(f"🔹 {selected_currency}")

    if selected_currency not in model_fits_signifikan:
        st.warning(f"Model ARIMA untuk {selected_currency} tidak tersedia.")
        st.stop()

    # Ambil parameter NGARCH (manual per mata uang)
    if selected_currency == 'IDR':
        omega, alpha, beta, gamma = 8.7206e-07, 0.1000, 0.8800, 0.0102
        model_type = 'NGARCH(1,1)'
    elif selected_currency == 'MYR':
        omega, alpha1, alpha2, beta, gamma = 7.9481e-07, 0.0906, 0.0905, 0.7300, 0.0382
        model_type = 'NGARCH(2,1)'
    elif selected_currency == 'SGD':
        omega, alpha, beta, gamma = 1.5322e-07, 0.0500, 0.9300, -0.0146
        model_type = 'NGARCH(1,1)'
    else:
        st.error("Parameter NGARCH tidak tersedia untuk mata uang ini.")
        st.stop()

    # Gabungkan return train + test
    returns_all = pd.concat([train_data[selected_currency], test_data[selected_currency]]).reset_index(drop=True)
    T = len(returns_all)
    h = np.zeros(T)

    # Hitung varians h(t)
    if selected_currency == 'MYR':
        h[0:2] = np.var(train_data[selected_currency])
        for t in range(2, T):
            eps1 = returns_all.iloc[t - 1]
            eps2 = returns_all.iloc[t - 2]
            term1 = alpha1 * (eps1 - gamma * np.sqrt(h[t - 1]))**2
            term2 = alpha2 * (eps2 - gamma * np.sqrt(h[t - 2]))**2
            h[t] = omega + term1 + term2 + beta * h[t - 1]
    else:
        h[0] = np.var(train_data[selected_currency])
        for t in range(1, T):
            eps_prev = returns_all.iloc[t - 1]
            h[t] = omega + alpha * (eps_prev - gamma * np.sqrt(h[t - 1]))**2 + beta * h[t - 1]

    forecasted_vol = np.sqrt(h[-30:])
    forecast_return = model_fits_signifikan[selected_currency].forecast(steps=30).reset_index(drop=True)

    # Prediksi harga
    last_train_index = train_data[selected_currency].index[-1]
    last_price = df.loc[last_train_index, selected_currency]
    forecast_price = last_price * np.exp(np.cumsum(forecast_return))
    upper_band = forecast_price * np.exp(forecasted_vol)
    lower_band = forecast_price * np.exp(-forecasted_vol)

    test_index = test_data[selected_currency].index
    actual_price = df.loc[test_index, selected_currency].values[:30]

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

    # Evaluasi volatilitas
    realized_var = test_data[selected_currency][:30]**2
    forecasted_var = forecasted_vol**2
    rmse = np.sqrt(mean_squared_error(realized_var, forecasted_var))
    mae = mean_absolute_error(realized_var, forecasted_var)
    mape = np.mean(np.abs((realized_var - forecasted_var) / realized_var)) * 100

    st.write("📊 **Evaluasi Prediksi Volatilitas (Squared Return vs Variance):**")
    st.write(f"**RMSE:** {rmse:.6f}")
    st.write(f"**MAE:** {mae:.6f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.markdown("---")
