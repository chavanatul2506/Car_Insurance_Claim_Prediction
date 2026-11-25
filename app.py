import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the three files saved by the notebook
model = joblib.load("claim_model.pkl")
model_columns = joblib.load("model_columns.pkl")
model_options = joblib.load("model_options.pkl")

# The probability above which we flag a policy as higher risk.
# 0.5 is the default; change it using the threshold table in the notebook (Step 10.4).
THRESHOLD = 0.5

st.set_page_config(page_title="Insurance Claim Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Insurance Claim Risk Predictor")
st.write("Fill in the policy and vehicle details below to estimate the probability "
         "that this policyholder will file a claim.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    subscription_length = st.slider("Subscription length (years)", 0.0, 14.0, 5.7, step=0.1)
    vehicle_age = st.slider("Vehicle age (years)", 0.0, 20.0, 1.2, step=0.1)
    customer_age = st.slider("Customer age", 35, 75, 44)
    region_density = st.number_input("Region population density", min_value=0, max_value=80000,
                                     value=8794, step=100)

with col2:
    vehicle_model = st.selectbox("Vehicle model", model_options)
    is_parking_sensors = st.checkbox("Has parking sensors", value=True)
    is_front_fog_lights = st.checkbox("Has front fog lights", value=True)
    is_brake_assist = st.checkbox("Has brake assist", value=True)
    is_power_steering = st.checkbox("Has power steering", value=True)
    is_driver_seat_height_adjustable = st.checkbox("Driver seat height adjustable", value=True)
    is_day_night_rear_view_mirror = st.checkbox("Day/night rear-view mirror", value=True)
    is_speed_alert = st.checkbox("Has speed alert", value=True)

st.divider()

if st.button("Predict claim risk", type="primary"):

    # 1. gather the inputs (checkbox True/False -> 1/0)
    input_dict = {
        "subscription_length": subscription_length,
        "vehicle_age": vehicle_age,
        "customer_age": customer_age,
        "region_density": region_density,
        "is_parking_sensors": int(is_parking_sensors),
        "is_front_fog_lights": int(is_front_fog_lights),
        "is_brake_assist": int(is_brake_assist),
        "is_power_steering": int(is_power_steering),
        "is_driver_seat_height_adjustable": int(is_driver_seat_height_adjustable),
        "is_day_night_rear_view_mirror": int(is_day_night_rear_view_mirror),
        "is_speed_alert": int(is_speed_alert),
    }

    # 2. one-row table
    input_df = pd.DataFrame([input_dict])

    # 3. one-hot encode the chosen car model by hand
    for col in model_columns:
        if col.startswith("model_"):
            input_df[col] = 1 if col == f"model_{vehicle_model}" else 0

    # 4. same columns, same order as in training
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 5. predict
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.metric("Estimated claim risk score", f"{proba * 100:.1f}%")
    st.caption("This is a relative risk score (the model was trained with class balancing), "
               "not an exact probability. Use it to compare policies.")

    if proba >= THRESHOLD:
        st.error("⚠️ This policy is flagged as HIGHER risk of a claim.")
    else:
        st.success("✅ This policy is flagged as LOWER risk of a claim.")

    st.progress(min(max(float(proba), 0.0), 1.0))
