import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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
    for col in ['Zone', 'Category', 'District']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category', 'District']
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
st.write("Enter details to predict monthly electricity usage (kWh).")

# Load data and train models
df = load_data()
models, label_encoders = train_models(df)

# User inputs
connected_load = st.number_input("Connected Load (kW)", min_value=0.0, value=10.0)

zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
district = st.selectbox("Select District", label_encoders['District'].classes_)
month = st.selectbox("Select Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

# Encode inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]
district_enc = label_encoders['District'].transform([district])[0]

# Predict
input_data = np.array([[connected_load, zone_enc, category_enc, district_enc]])
prediction = models[month].predict(input_data)[0]

# Show result
st.success(f"📊 Predicted electricity consumption for **{month}**: **{prediction:.2f} kWh**")
