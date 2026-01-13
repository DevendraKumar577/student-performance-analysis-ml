import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(
    page_title="Student Performance Risk Predictor",
    layout="centered"
)

st.title("Student Performance Risk Predictor")
st.write("Predict student academic risk using a Machine Learning model")
st.divider()


@st.cache_resource
def load_artifacts():
    model = joblib.load("student_risk_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, label_encoder, feature_names

model, label_encoder, feature_names = load_artifacts()

st.subheader("Enter Student Academic Details")

quiz_avg = st.slider("Quiz Average (%)", 0, 100, 70)
assignment_avg = st.slider("Assignment Average (%)", 0, 100, 85)
midterm_score = st.slider("Midterm Exam Score (%)", 0, 100, 85)
final_exam_score = st.slider("Final Exam Score (%)", 0, 100, 90)

display_df = pd.DataFrame([{
    "Quiz Average (%)": quiz_avg,
    "Assignment Average (%)": assignment_avg,
    "Midterm Exam Score (%)": midterm_score,
    "Final Exam Score (%)": final_exam_score
}])

model_input = pd.DataFrame([{
    "Quiz_Average": quiz_avg,
    "Assignment_Average": assignment_avg,
    "Midterm_Score": midterm_score,
    "Final_Exam_Score": final_exam_score
}])

# Align with training features (CRITICAL FOR ML)
model_input = model_input.reindex(columns=feature_names, fill_value=0)

if st.button("Predict Risk"):

    prediction = model.predict(model_input)[0]
    result = label_encoder.inverse_transform([prediction])[0]

    st.divider()
    st.subheader("Prediction Result")

    if result == "At Risk":
        st.error("Student is AT RISK")
    elif result == "Medium":
        st.warning("Student is at MEDIUM RISK")
    else:
        st.success("Student is a HIGH PERFORMER")

    st.subheader("Student Input Summary")
    st.dataframe(display_df, use_container_width=True)

    st.info(
        "Note: The prediction is based on academic performance indicators. "
        "Features not provided by the user are handled internally to maintain "
        "model consistency."
    )















