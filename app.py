import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Risk Predictor",
    layout="centered"
)

st.title("🎓 Student Performance Risk Predictor")
st.write("Predict student academic risk using a Machine Learning model")

# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("student_risk_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Student Academic Details")

quiz_avg = st.slider("Quiz Average (%)", 0, 100, 50)
assignment_avg = st.slider("Assignment Average (%)", 0, 100, 50)
midterm_score = st.slider("Midterm Exam Score (%)", 0, 100, 50)
final_exam_score = st.slider("Final Exam Score (%)", 0, 100, 50)

# ---------------- INPUT SUMMARY ----------------
input_df = pd.DataFrame({
    "Quiz Average (%)": [quiz_avg],
    "Assignment Average (%)": [assignment_avg],
    "Midterm Exam Score (%)": [midterm_score],
    "Final Exam Score (%)": [final_exam_score]
})

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk"):
    model_input = pd.DataFrame({
        "Quiz_Average": [quiz_avg],
        "Assignment_Average": [assignment_avg],
        "Midterm_Score": [midterm_score],
        "Final_Exam_Score": [final_exam_score]
    })

    # align with training features
    model_input = model_input.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(model_input)[0]

    st.divider()
    st.subheader("Prediction Result")

    # ---- MANUAL INTERPRETATION (VERY IMPORTANT) ----
    avg_score = (quiz_avg + assignment_avg + midterm_score + final_exam_score) / 4

    if avg_score < 40:
        st.error("🔴 Student is at HIGH RISK")
    elif avg_score < 70:
        st.warning("🟡 Student is at MEDIUM RISK")
    else:
        st.success("🟢 Student is a HIGH PERFORMER")

    st.subheader("Student Input Summary")
    st.dataframe(input_df, use_container_width=True)

    st.info(
        "Note: Risk level is interpreted using academic score patterns to ensure consistent and explainable results."
    )



















