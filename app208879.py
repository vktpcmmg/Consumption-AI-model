import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import base64

def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_image_base64("tata_logo.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<h1 style='text-align: center; color: #003366;'>⚡3 Phase Power and Energy Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>",
    unsafe_allow_html=True
)
# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv("consumptionai.csv")  # File must be in the same GitHub repo
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    return df

# Train models and label encoders
@st.cache_resource
def train_models(df):
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    for month in months:
        X = df[input_features]
        y = df[month]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# Streamlit app layout
st.title("🔌 Electricity Consumption Predictor")

st.markdown("*_Note: This is based on around 90K smart meter data from FY 24–25._*")
st.write("Enter details to predict monthly electricity usage (kWh).")

st.markdown("*_Note: This is based on around 90K smart meter data from FY 24–25._*")

# Load data and train models
df = load_data()
models, label_encoders = train_models(df)

# User inputs
connected_load = st.number_input("Connected Load (kW)", min_value=0.0, value=10.0)

zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
month = st.selectbox("Select Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

# Encode inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

if st.button("🔍 Predict Consumption"):
    input_data = np.array([[connected_load, zone_enc, category_enc]])

    # Predict for selected month
    selected_month_prediction = models[month].predict(input_data)[0]
    st.success(f"📊 Predicted electricity consumption for **{month}**: **{selected_month_prediction:.2f} kWh**")

    # Predict for all months
    months = list(models.keys())
    monthly_predictions = {m: models[m].predict(input_data)[0] for m in months}

    # Encode columns to match with dataset
    df_encoded = df.copy()
    for col, le in label_encoders.items():
        df_encoded[col] = le.transform(df_encoded[col])

    # Find the best match row
    matched_row = df_encoded[
        (df_encoded['Zone'] == zone_enc) &
        (df_encoded['Category'] == category_enc)
    ]
    if matched_row.empty:
        st.warning("No matching row found for actual comparison. Showing only predicted values.")
        actual_values = {m: None for m in months}
    else:
        # Get row with closest connected load
        matched_row = matched_row.iloc[(matched_row['Connected Load'] - connected_load).abs().argsort()[:1]]
        actual_values = matched_row[months].iloc[0].to_dict()

    # Build DataFrame for comparison
    compare_df = pd.DataFrame({
        'Month': months,
        'Predicted (kWh)': [monthly_predictions[m] for m in months],
        'Actual (kWh)': [actual_values[m] if actual_values[m] is not None else np.nan for m in months]
    })

    compare_df.set_index('Month', inplace=True)

    # Show chart
    st.subheader("📊 Actual vs Predicted Monthly Consumption")
    st.line_chart(compare_df)
