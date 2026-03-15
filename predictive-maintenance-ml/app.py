from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"

model = joblib.load(MODEL_PATH)

st.title("Predictive Maintenance Model")
st.write("Predict whether industrial equipment is at risk of failure.")

machine_type = st.selectbox("Machine Type", ["Type A", "Type B", "Type C"])
air_temperature = st.number_input("Air Temperature", value=300.0)
process_temperature = st.number_input("Process Temperature", value=310.0)
rotational_speed = st.number_input("Rotational Speed", value=1500.0)
torque = st.number_input("Torque", value=40.0)
tool_wear = st.number_input("Tool Wear", value=100)
operating_hours = st.number_input("Operating Hours", value=3500.0)
vibration = st.number_input("Vibration", value=2.5)
pressure = st.number_input("Pressure", value=55.0)

if st.button("Predict Failure"):
    input_df = pd.DataFrame({
        "machine_type": [machine_type],
        "air_temperature": [air_temperature],
        "process_temperature": [process_temperature],
        "rotational_speed": [rotational_speed],
        "torque": [torque],
        "tool_wear": [tool_wear],
        "operating_hours": [operating_hours],
        "vibration": [vibration],
        "pressure": [pressure],
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write("Failure Prediction:", int(prediction))
    st.write("Failure Probability:", round(float(probability), 4))


