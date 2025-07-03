import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and expected features
model, feature_names = pickle.load(open("car_model.pkl", "rb"))

# Session reset
def reset_inputs():
    for key in st.session_state.keys():
        del st.session_state[key]

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")

with st.form("form"):
    st.subheader("🛠 Enter Car Details")

    col1, col2 = st.columns(2)
    with col1:
        year = st.slider("Year of Manufacture", 1995, 2025, 2015)
        present_price = st.number_input("Present Price (in Lakhs)", 0.0, 50.0, 5.0)
        kms_driven = st.number_input("Kilometers Driven", 0, 300000, 30000)

    with col2:
        owner = st.selectbox("Number of Owners", [0, 1, 2, 3])
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

    submitted = st.form_submit_button("🔍 Predict")
    reset = st.form_submit_button("🔄 Reset", on_click=reset_inputs)

if submitted:
    # One-hot encoding manually
    input_dict = {
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Owner': owner,
        'Year': year,
        'Fuel_Diesel': 1 if fuel_type == 'Diesel' else 0,
        'Fuel_Petrol': 1 if fuel_type == 'Petrol' else 0,
        'Seller_Type_Individual': 1 if seller_type == 'Individual' else 0,
        'Transmission_Manual': 1 if transmission == 'Manual' else 0
    }

    # Build DataFrame from input_dict, reordering columns to match model
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names)

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹{prediction:.2f} Lakhs")
