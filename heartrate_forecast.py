import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----- Setup -----
st.set_page_config(page_title="Heart Rate Forecast", layout="wide")
st.title("❤️ Heart Rate Forecasting with ARIMA & Holt-Winters")

# ----- Helper: Unique Columns -----
def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        col = col.strip()
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

# ----- File Upload -----
uploaded_file = st.file_uploader("📤 Upload your heart rate CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = make_unique_columns(df.columns)

    st.subheader("👀 Data Preview")
    st.dataframe(df.head())

    time_col = st.selectbox("🕒 Select Timestamp Column", df.columns)
    value_col = st.selectbox("❤️ Select Heart Rate Column", df.columns)

    # ----- Clean Data -----
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df[[time_col, value_col]].dropna()
    df = df.sort_values(by=time_col)
    df.set_index(time_col, inplace=True)

    st.subheader("🧹 Data Cleaning")
    max_hr = st.slider("Max Heart Rate Threshold", 100, 250, 200)
    df = df[df[value_col] <= max_hr]
    df[value_col] = df[value_col].interpolate(method='time')
    st.line_chart(df[value_col], use_container_width=True)

    # ----- Forecast Settings -----
    st.subheader("🔮 Forecast Configuration")
    model_type = st.selectbox("Choose Model", ["ARIMA", "Holt-Winters"])
    forecast_periods = st.number_input("Future time steps to forecast", 10, 500, 30)
    test_size = st.slider("Test size (for evaluation)", 10, 200, 30)

    # ----- ARIMA Tuning -----
    if model_type == "ARIMA":
        st.markdown("📌 **ARIMA(p, d, q) Parameters**")
        p = st.number_input("p (autoregressive)", 0, 10, 5)
        d = st.number_input("d (difference)", 0, 2, 1)
        q = st.number_input("q (moving average)", 0, 10, 0)

    # ----- Train/Test Split -----
    series = df[value_col]
    train, test = series[:-test_size], series[-test_size:]

    with st.spinner("Training model and forecasting..."):
        try:
            if model_type == "ARIMA":
                # Train ARIMA
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                test_forecast = model_fit.forecast(steps=test_size)
                rmse = np.sqrt(mean_squared_error(test, test_forecast))
                mape = mean_absolute_percentage_error(test, test_forecast) * 100

                # Final model and forecast
                final_model = ARIMA(series, order=(p, d, q)).fit()
                forecast_result = final_model.get_forecast(steps=forecast_periods)
                forecast_mean = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()

                future_index = pd.date_range(start=series.index[-1] + timedelta(minutes=1),
                                             periods=forecast_periods, freq='T')
                forecast_df = pd.DataFrame({
                    f"{value_col}_Forecast": forecast_mean.values,
                    "lower_ci": conf_int.iloc[:, 0].values,
                    "upper_ci": conf_int.iloc[:, 1].values
                }, index=future_index)

            else:
                # Train Holt-Winters
                model = ExponentialSmoothing(train, trend="add", seasonal=None)
                model_fit = model.fit()
                test_forecast = model_fit.forecast(test_size)
                rmse = np.sqrt(mean_squared_error(test, test_forecast))
                mape = mean_absolute_percentage_error(test, test_forecast) * 100

                # Final Holt model
                final_model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
                forecast_values = final_model.forecast(forecast_periods)

                future_index = pd.date_range(start=series.index[-1] + timedelta(minutes=1),
                                             periods=forecast_periods, freq='T')
                forecast_df = pd.DataFrame({
                    f"{value_col}_Forecast": forecast_values.values
                }, index=future_index)

            # ----- Evaluation -----
            st.subheader("📊 Evaluation")
            st.write(f"**RMSE**: {rmse:.2f}")
            st.write(f"**MAPE**: {mape:.2f}%")

            # ----- Plotting -----
            st.subheader("📈 Forecast Plot")
            fig, ax = plt.subplots(figsize=(10, 5))

            # Plot training data
            train.plot(ax=ax, label="Training", color="green", linewidth=2)

            # Plot test data
            test.plot(ax=ax, label="Test", color="red", linewidth=2)

            # Plot forecast (future)
            forecast_df[f"{value_col}_Forecast"].plot(
                ax=ax, label="Forecast", color="blue", linestyle="--", linewidth=2.5
            )

            # Optional: Add ARIMA confidence interval
            if model_type == "ARIMA" and "lower_ci" in forecast_df.columns:
                ax.fill_between(
                    forecast_df.index,
                    forecast_df["lower_ci"],
                    forecast_df["upper_ci"],
                    color="blue", alpha=0.2, label="Confidence Interval"
                )

            # Style
            ax.set_title(f"{model_type} Forecast", fontsize=16)
            ax.set_xlabel("Time")
            ax.set_ylabel("Heart Rate")
            ax.axvline(series.index[-1], color="gray", linestyle=":", linewidth=1.5)
            ax.grid(True)
            ax.legend()

            st.pyplot(fig)

            # ----- Download -----
            st.subheader("⬇ Download Forecast Data")
            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button("Download CSV", csv, file_name="forecast.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error during forecasting: {e}")
