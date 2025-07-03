import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler

# Set custom page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide", page_icon="📊")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .title {
            font-size: 36px;
            color: #d62828;
            text-align: center;
            font-weight: bold;
        }
        .subtext {
            font-size: 16px;
            text-align: center;
            color: #444;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: gray;
            margin-top: 20px;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 180px;
            margin-top: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
working_dir = os.path.dirname(os.path.abspath(__file__))
performance_model = pickle.load(open(f"{working_dir}/student_performance_model.sav", 'rb'))

# Load logo
st.markdown("<img class='logo' src='https://raw.githubusercontent.com/Prasad777777/CSI/main/Final_Project/celebal_logo.png'>", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='title'>Student Performance Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>An AI-powered tool developed during internship at Celebal Technologies</div>", unsafe_allow_html=True)

# Author credit
st.markdown("<div class='footer'>Created by <strong>Prasad Baban Parjane</strong> | <a href='https://github.com/Prasad777777' target='_blank'>GitHub</a></div>", unsafe_allow_html=True)

# Form layout
st.markdown("<div class='main'>", unsafe_allow_html=True)
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
    tutoring_sessions = st.selectbox("📚 Tutoring Sessions", list(range(9)))
    family_income = st.selectbox("💰 Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("👩‍🏫 Teacher Quality", ["Low", "Medium", "High"])

with col3:
    school_type = st.selectbox("🏫 School Type", ["Public", "Private"])
    peer_influence = st.selectbox("👫 Peer Influence", ["Negative", "Neutral", "Positive"])
    physical_activity = st.selectbox("🏃 Physical Activity (Hours)", list(range(7)))
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

# Scaling
scale = lambda v, mi, ma: (v - mi) / (ma - mi)
data['Attendance'] = scale(data['Attendance'][0], 60, 100)
data['Hours_Studied'] = scale(data['Hours_Studied'][0], 1, 44)
data['Previous_Scores'] = scale(data['Previous_Scores'][0], 50, 100)
data['Sleep_Hours'] = scale(data['Sleep_Hours'][0], 4, 10)
data['Tutoring_Sessions'] = scale(data['Tutoring_Sessions'][0], 0, 8)
data['Physical_Activity'] = scale(data['Physical_Activity'][0], 0, 6)

# Encoding
enc_map = {
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
for col, mapping in enc_map.items():
    data[col] = data[col].map(mapping)

# Prediction
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🔍 Prediction Result")
if st.button("🚀 Predict"):
    prediction = performance_model.predict(data)
    st.success(f"🎯 Predicted Student Performance Score: **{prediction[0]:.2f}**")

    # 📊 Bar chart of scaled numeric inputs
    st.write("### 📊 Scaled Input Features")
    fig, ax = plt.subplots(figsize=(10, 4))
    numeric_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
    sns.barplot(x=numeric_cols, y=data[numeric_cols].iloc[0], palette='Reds', ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Encoded categorical table
    st.write("### 🧩 Encoded Categorical Features")
    st.dataframe(data.drop(columns=numeric_cols).T.rename(columns={0: "Encoded Value"}))
