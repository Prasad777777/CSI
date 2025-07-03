import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="wide", page_icon="📊")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .author-info {
            font-size: 15px;
            color: #777;
            margin-top: -10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Author credit
st.markdown("""
    <p class='author-info'>
        Created by <strong>Prasad Baban Parjane</strong> ✨ |
        <a href='https://github.com/Prasad777777' target='_blank'>GitHub</a>
    </p>
""", unsafe_allow_html=True)

# Load model
working_dir = os.path.dirname(os.path.abspath(__file__))
performance_model = pickle.load(open(f"{working_dir}/student_performance_model.sav", 'rb'))

# Title
st.title("📚 Smart Student Performance Predictor")
st.markdown("Fill in the details below and get a smart prediction of the student's expected performance score! 🧠")

# Form layout
st.markdown("---")
col1, col2, col3 = st.columns(3)

# Inputs - col1
with col1:
    hours_studied = st.number_input("Hours Studied", min_value=1, max_value=44, step=1)
    attendance = st.slider("Attendance (%)", min_value=60, max_value=100, step=1)
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    sleep_hours = st.selectbox("Sleep Hours", [4, 5, 6, 7, 8, 9, 10])

# Inputs - col2
with col2:
    previous_scores = st.slider("Previous Scores (%)", min_value=50, max_value=100, step=1)
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    tutoring_sessions = st.selectbox("Tutoring Sessions", list(range(9)))
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])

# Inputs - col3
with col3:
    school_type = st.selectbox("School Type", ["Public", "Private"])
    peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    physical_activity = st.selectbox("Physical Activity (Hours)", list(range(7)))
    learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
    parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Preprocess input
features = pd.DataFrame({
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

ordered_columns = features.columns.tolist()

# Scaling
def scale(val, min_, max_): return (val - min_) / (max_ - min_)
features['Attendance'] = scale(features['Attendance'][0], 60, 100)
features['Hours_Studied'] = scale(features['Hours_Studied'][0], 1, 44)
features['Previous_Scores'] = scale(features['Previous_Scores'][0], 50, 100)
features['Sleep_Hours'] = scale(features['Sleep_Hours'][0], 4, 10)
features['Tutoring_Sessions'] = scale(features['Tutoring_Sessions'][0], 0, 8)
features['Physical_Activity'] = scale(features['Physical_Activity'][0], 0, 6)

# Encoding
mapping = {
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

for col, m in mapping.items():
    features[col] = features[col].map(m)

# Prediction
st.markdown("---")
st.subheader("🎯 Prediction Result")
if st.button("🔍 Predict Performance"):
    result = performance_model.predict(features)
    st.success(f"Predicted Score: **{result[0]:.2f}**")

    # Visuals
    st.subheader("📊 Scaled Feature Impact")
    fig, ax = plt.subplots(figsize=(10, 4))
    numeric_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
    sns.barplot(x=numeric_cols, y=features[numeric_cols].iloc[0], palette="coolwarm", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Scaled Value")
    st.pyplot(fig)

    st.subheader("🧩 Encoded Categorical Values")
    cat_data = features.drop(columns=numeric_cols).T.rename(columns={0: "Encoded Value"})
    st.dataframe(cat_data)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
