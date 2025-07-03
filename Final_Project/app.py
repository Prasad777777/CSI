import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set custom theme and logo
st.set_page_config(
    page_title="Student Performance Predictor - Celebal Internship",
    layout="wide",
    page_icon="📊"
)

# Logo & Header
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            color: #B22222;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            margin-top: 30px;
            color: #888;
        }
        .stApp {
            background-color: #fff8f8;
        }
    </style>
    <img src="https://raw.githubusercontent.com/Prasad777777/CSI/main/Final_Project/download.jpeg" class="logo" />
    <h1 class='main-title'>🎓 Student Performance Predictor</h1>
    <p style='text-align:center;'>An Internship Project by <strong>Prasad Baban Parjane</strong> @ <a href='https://www.celebaltech.com' target='_blank'>Celebal Technologies</a> | <a href='https://github.com/Prasad777777' target='_blank'>GitHub</a></p>
    """,
    unsafe_allow_html=True
)

# Load the saved model
working_dir = os.path.dirname(os.path.abspath(__file__))
performance_model = pickle.load(open(f"{working_dir}/student_performance_model.sav", 'rb'))

st.write("Fill in the details below to predict the student's performance score.")

# Form layout
col1, col2, col3 = st.columns(3)

with col1:
    hours_studied = st.number_input("📖 Hours Studied", min_value=1, max_value=44, step=1)
    attendance = st.slider("🎓 Attendance (%)", min_value=60, max_value=100, step=1)
    parental_involvement = st.selectbox("👨‍👩‍👧 Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("📚 Access to Resources", ["Low", "Medium", "High"])
    extracurricular_activities = st.selectbox("🏆 Extracurricular Activities", ["No", "Yes"])
    sleep_hours = st.selectbox("💤 Sleep Hours", [4, 5, 6, 7, 8, 9, 10])

with col2:
    previous_scores = st.slider("📊 Previous Scores (%)", min_value=50, max_value=100, step=1)
    motivation_level = st.selectbox("💪 Motivation Level", ["Low", "Medium", "High"])
    internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"])
    tutoring_sessions = st.selectbox("📚 Tutoring Sessions", list(range(0, 9)))
    family_income = st.selectbox("💰 Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("👩‍🏫 Teacher Quality", ["Low", "Medium", "High"])

with col3:
    school_type = st.selectbox("🏫 School Type", ["Public", "Private"])
    peer_influence = st.selectbox("👫 Peer Influence", ["Negative", "Neutral", "Positive"])
    physical_activity = st.selectbox("🏃 Physical Activity (Hours)", list(range(0, 7)))
    learning_disabilities = st.selectbox("🧠 Learning Disabilities", ["No", "Yes"])
    parental_education = st.selectbox("🎓 Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home = st.selectbox("🏠 Distance from Home", ["Near", "Moderate", "Far"])
    gender = st.selectbox("👤 Gender", ["Male", "Female"])

# Data preparation
data = pd.DataFrame({
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Parental_Involvement': [parental_involvement],
    'Access_to_Resources': [access_to_resources],
    'Extracurricular_Activities': [extracurricular_activities],
    'Sleep_Hours': [sleep_hours],
    'Previous_Scores': [previous_scores],
    'Motivation_Level': [motivation_level],
    'Internet_Access': [internet_access],
    'Tutoring_Sessions': [tutoring_sessions],
    'Family_Income': [family_income],
    'Teacher_Quality': [teacher_quality],
    'School_Type': [school_type],
    'Peer_Influence': [peer_influence],
    'Physical_Activity': [physical_activity],
    'Learning_Disabilities': [learning_disabilities],
    'Parental_Education_Level': [parental_education],
    'Distance_from_Home': [distance_from_home],
    'Gender': [gender]
})

# Reorder
data = data[[
    'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
    'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
    'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]]

# Scaling
def custom_scale(value, min_v, max_v):
    return (value - min_v) / (max_v - min_v)

scaling = {
    'Attendance': (60, 100),
    'Hours_Studied': (1, 44),
    'Previous_Scores': (50, 100),
    'Sleep_Hours': (4, 10),
    'Tutoring_Sessions': (0, 8),
    'Physical_Activity': (0, 6)
}

for col, (min_val, max_val) in scaling.items():
    data[col] = custom_scale(data[col][0], min_val, max_val)

# Encoding
encodings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Internet_Access': {'Yes': 1, 'No': 0},
    'School_Type': {'Public': 1, 'Private': 0},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2}
}

for col, mapping in encodings.items():
    data[col] = data[col].map(mapping)

# Predict
st.write("### 🔍 Prediction Result")
if st.button("🚀 Predict"):
    prediction = performance_model.predict(data)
    st.success(f"🎯 Predicted Student Performance Score: **{prediction[0]:.2f}**")

    st.write("### 📊 Feature Breakdown")
    num_cols = list(scaling.keys())
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=num_cols, y=data[num_cols].iloc[0], palette='Reds', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Scaled Value")
    ax.set_title("Scaled Numeric Inputs")
    st.pyplot(fig)

    st.write("### 🧩 Categorical Feature Encoding")
    st.dataframe(data.drop(columns=num_cols).T.rename(columns={0: "Encoded Value"}))

# Footer
st.markdown(
    """
    <div class='footer'>
        🚀 Powered by <strong>Celebal Technologies</strong> | Internship Project 2025
    </div>
    """,
    unsafe_allow_html=True
)
