🫀 Heart Rate Forecasting using Time Series Models
🚀 Why?
Continuous heart rate monitoring can provide early signs of health risks or recovery patterns. But beyond observing the past, predicting future heart rate trends can help:

Anticipate abnormal spikes or drops

Manage post-operative care

Improve athlete performance tracking

Enable smarter healthcare alerts

This project aims to provide an easy-to-use web app for forecasting heart rate using time series modeling.

💡 What?
An interactive Streamlit application that allows users to:

✅ Upload heart rate CSV data
✅ Clean and preprocess data (handle outliers, interpolate missing)
✅ Choose between ARIMA or Holt-Winters forecasting models
✅ Tune ARIMA parameters (p, d, q)
✅ Visualize:

Training vs. Test vs. Forecast data

Forecast confidence intervals (for ARIMA)
✅ Evaluate model using RMSE and MAPE
✅ Download forecasted results as a CSV

⚙️ How?
🔧 Requirements
bash
Copy
Edit
pip install streamlit pandas numpy matplotlib statsmodels scikit-learn
▶️ Run the App
bash
Copy
Edit
streamlit run heartrate_forecast.py
📂 Input Format
Upload a .csv file with at least:

A timestamp column (e.g., Timestamp (GMT))

A heart rate column (e.g., Lifetouch Heart Rate)

Example:

python-repl
Copy
Edit
Timestamp (GMT),Lifetouch Heart Rate
2025-06-23 14:00:00,78
2025-06-23 14:01:00,80
...
🖼️ Example Forecast Output
The app separates:

Green: Training data

Red: Test (evaluation) data

Blue: Future forecast

Shaded area: ARIMA confidence interval

🧠 Models
ARIMA: Good for data with clear trends but no seasonality

Holt-Winters (Exponential Smoothing): Good for data with gradual trends and smoother transitions

Both models evaluated on last n points using:

RMSE: Root Mean Squared Error

MAPE: Mean Absolute Percentage Error

📦 Output
Forecasted values downloadable as .csv

All results visible inline in the Streamlit app

📌 Roadmap Ideas
 Add interactive Plotly visualizations

 Add SARIMA / seasonal models

 Incorporate anomaly detection

 Add real-time data feed support (e.g. via API or device)

🙌 Contributions
Feel free to fork, improve the models, or add support for more sensors. PRs welcome!


find the link: http://localhost:8501/
