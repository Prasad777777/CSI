import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie

# Set page configuration
st.set_page_config(page_title="Student Exam Score Prediction", layout="wide", page_icon="📊")

# Load Lottie animation from working URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

celebal_lottie = load_lottieurl("https://lottie.host/4c39b5a0-8ef9-4b5e-bad6-614ce2691181/zYFqU48fL9.json")

# Load Celebal logo
working_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(working_dir, "download.jpeg")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=220)

# Add Lottie animation if available
if celebal_lottie:
    st_lottie(celebal_lottie, height=200, key="celebal")
else:
    st.warning("⚠️ Unable to load animation. Please check the Lottie URL.")

# Custom CSS for red/white Celebal theme
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
    }
    .main {
        background-color: #ffffff;
        color: #8B0000;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stButton>button {
        background-color: #8B0000;
        color: white;
    }
    .author-info {
        font-size: 14px;
        color: gray;
    }
    </style>
    <p class='author-info'>Created by <strong>Prasad Baban Parjane</strong> | <a href='https://github.com/Prasad777777' target='_blank'>GitHub</a></p>
    """,
    unsafe_allow_html=True
)

# Load model
performance_model = pickle.load(open(f"{working_dir}/student_performance_model.sav", 'rb'))

# Title and description
st.title("🎓 Student Exam Score Prediction")
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

# Data preprocessing
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

ordered_columns = list(data.columns)
data = data[ordered_columns]

# Custom scaling
def custom_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

data['Attendance'] = custom_scale(data['Attendance'][0], 60, 100)
data['Hours_Studied'] = custom_scale(data['Hours_Studied'][0], 1, 44)
data['Previous_Scores'] = custom_scale(data['Previous_Scores'][0], 50, 100)
data['Sleep_Hours'] = custom_scale(data['Sleep_Hours'][0], 4, 10)
data['Tutoring_Sessions'] = custom_scale(data['Tutoring_Sessions'][0], 0, 8)
data['Physical_Activity'] = custom_scale(data['Physical_Activity'][0], 0, 6)

# Label encoding
mappings = {
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

for col, mapping in mappings.items():
    data[col] = data[col].map(mapping)

# Prediction
st.write("### 🔍 Prediction Result")
if st.button("🚀 Predict"):
    prediction = performance_model.predict(data)
    st.success(f"🎯 The predicted student performance score is: **{prediction[0]:.2f}**")

    # 📊 Bar chart of scaled numeric inputs
    st.write("### 📊 Input Feature Distribution")
    numeric_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
    scaled_data = data[numeric_features]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=scaled_data.columns, y=scaled_data.iloc[0], palette='Reds', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Scaled Value (0 to 1)")
    ax.set_title("Impact of Scaled Numerical Features")
    st.pyplot(fig)

    # 📋 Encoded Categorical Features Table
    st.write("### 🧩 Encoded Categorical Features")
    categorical_data = data.drop(columns=numeric_features)
    st.dataframe(categorical_data.T.rename(columns={0: "Encoded Value"}))
