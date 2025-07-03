import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

INR_TO_USD = 0.012  # 1 INR ≈ $0.012
INR_TO_UGX = 45.0   # 1 INR ≈ 45 UGX

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

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names)

        st.write("🧪 Input DataFrame (for debugging):", input_df)  # Optional debug line

        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Selling Price: ₹{prediction:.2f} Lakhs")


if submitted:
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

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names)

    # Show input for debugging
    st.write("🧪 Input DataFrame (for debugging):", input_df)

    # Predict in INR
    prediction_inr = model.predict(input_df)[0]
    prediction_usd = prediction_inr * 100000 * INR_TO_USD
    prediction_ugx = prediction_inr * 100000 * INR_TO_UGX

    # Show prediction results
    st.subheader("💰 Estimated Selling Price")
    st.write(f"- 🇮🇳 INR: ₹{prediction_inr:.2f} Lakhs")
    st.write(f"- 🇺🇸 USD: ${prediction_usd:,.2f}")
    st.write(f"- 🇺🇬 UGX: USh {prediction_ugx:,.0f}")

    # Save input + all currency predictions to file
    log_df = input_df.copy()
    log_df["Predicted_INR_Lakhs"] = prediction_inr
    log_df["Predicted_USD"] = prediction_usd
    log_df["Predicted_UGX"] = prediction_ugx

    log_df.to_csv("user_prediction_log.csv", mode='a', header=not os.path.exists("user_prediction_log.csv"), index=False)
    st.success("✅ Prediction saved to 'user_prediction_log.csv'")


    




