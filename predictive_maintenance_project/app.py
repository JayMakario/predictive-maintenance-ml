
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.title("Predictive Maintenance Model")

machine_type = st.selectbox("Machine Type",["Type A","Type B","Type C"])
air_temperature = st.number_input("Air Temperature")
process_temperature = st.number_input("Process Temperature")
rotational_speed = st.number_input("Rotational Speed")
torque = st.number_input("Torque")
tool_wear = st.number_input("Tool Wear")
operating_hours = st.number_input("Operating Hours")
vibration = st.number_input("Vibration")
pressure = st.number_input("Pressure")

if st.button("Predict Failure"):
    input_df = pd.DataFrame({
        "machine_type":[machine_type],
        "air_temperature":[air_temperature],
        "process_temperature":[process_temperature],
        "rotational_speed":[rotational_speed],
        "torque":[torque],
        "tool_wear":[tool_wear],
        "operating_hours":[operating_hours],
        "vibration":[vibration],
        "pressure":[pressure]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write("Failure Prediction:", prediction)
    st.write("Failure Probability:", probability)
