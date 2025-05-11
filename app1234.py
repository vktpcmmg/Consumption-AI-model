import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import requests

# Page config
st.set_page_config(page_title="Smart Meter Consumption Predictor", layout="centered")

# Tata Power Logo
st.image("https://upload.wikimedia.org/wikipedia/en/2/27/Tata_Power_Logo.png", width=150)
st.markdown("### Designed by Tata Power MMG")

# Italic note
st.markdown("*This is based on around 90K smart meter data from FY 24–25.*")

# Load data
csv_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/consumptionai.csv"
df = pd.read_csv(csv_url)

# Drop 'District'
if 'District' in df.columns:
    df = df.drop(columns=['District'])

# Clean column names
df.columns = df.columns.str.strip()

# Input features
input_features = ['Connected Load', 'Zone', 'Category']

# Encode categorical columns
label_encoders = {}
for col in ['Zone', 'Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train separate models for each month
months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
models = {}

for month in months:
    X = df[input_features]
    y = df[month]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[month] = model

# User inputs
st.header("📈 Monthly Consumption Prediction")

connected_load = st.number_input("Connected Load (kW)", step=0.1)
zone = st.selectbox("Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Category", label_encoders['Category'].classes_)
month = st.selectbox("Select Month", months)

zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

# Validate load
if connected_load <= 0:
    st.error("⚠️ Please enter valid load as Zero load and Negative load don't exist.")
else:
    if st.button("🔍 Predict Consumption"):
        input_data = np.array([[connected_load, zone_enc, category_enc]])

        # Predict for selected month
        selected_month_prediction = models[month].predict(input_data)[0]
        st.success(f"📊 Predicted electricity consumption for **{month}**: **{selected_month_prediction:.2f} kWh**")

        # Predict for all months
        monthly_predictions = {m: models[m].predict(input_data)[0] for m in months}

        # Find closest matching row for actual
        df_encoded = df.copy()
        for col, le in label_encoders.items():
            df_encoded[col] = le.transform(df_encoded[col])

        matched_row = df_encoded[
            (df_encoded['Zone'] == zone_enc) &
            (df_encoded['Category'] == category_enc)
        ]
        if matched_row.empty:
            st.warning("No matching row found for actual comparison. Showing only predicted values.")
            actual_values = {m: None for m in months}
        else:
            matched_row = matched_row.iloc[(matched_row['Connected Load'] - connected_load).abs().argsort()[:1]]
            actual_values = matched_row[months].iloc[0].to_dict()

        # Prepare comparison DataFrame
        compare_df = pd.DataFrame({
            'Month': months,
            'Predicted (kWh)': [monthly_predictions[m] for m in months],
            'Actual (kWh)': [actual_values[m] if actual_values[m] is not None else np.nan for m in months]
        })
        compare_df.set_index('Month', inplace=True)

        # Plot
        st.subheader("📊 Actual vs Predicted Monthly Consumption")
        st.line_chart(compare_df)
