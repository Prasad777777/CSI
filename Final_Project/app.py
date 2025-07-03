import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Student Exam Score Prediction", layout="wide", page_icon="📊")

# Add author and GitHub link
st.markdown(
    """
    <style>
        .author-info {
            font-size: 14px;
            color: gray;
        }
    </style>
    <p class='author-info'>Created by <strong>Prasad Baban Parjane</strong> | <a href='https://github.com/Prasad777777' target='_blank'>GitHub</a></p>
    """,
    unsafe_allow_html=True
)

# Load the saved model
working_dir = os.path.dirname(os.path.abspath(__file__))
performance_model = pickle.load(open(f"{working_dir}/student_performance_model.sav", 'rb'))

# Title and description
st.title("🎓 Student Performance Predictor")
st.write("Fill in the details below to predict the student's performance score.")

# Create form layout using columns
col1, col2, col3 = st.columns(3)

# Input fields in col1
with col1:
    hours_studied = st.number_input("📖 Hours Studied", min_value=1, max_value=44, step=1)
    attendance = st.slider("🎓 Attendance (%)", min_value=60, max_value=100, step=1)
    parental_involvement = st.selectbox("👨‍👩‍👧 Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("📚 Access to Resources", ["Low", "Medium", "High"])
    extracurricular_activities = st.selectbox("🏆 Extracurricular Activities", ["No", "Yes"])
    sleep_hours = st.selectbox("💤 Sleep Hours", [4, 5, 6, 7, 8, 9, 10])

# Input fields in col2
with col2:
    previous_scores = st.slider("📊 Previous Scores (%)", min_value=50, max_value=100, step=1)
    motivation_level = st.selectbox("💪 Motivation Level", ["Low", "Medium", "High"])
    internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"])
    tutoring_sessions = st.selectbox("📚 Tutoring Sessions", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    family_income = st.selectbox("💰 Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("👩‍🏫 Teacher Quality", ["Low", "Medium", "High"])

# Input fields in col3
with col3:
    school_type = st.selectbox("🏫 School Type", ["Public", "Private"])
    peer_influence = st.selectbox("👫 Peer Influence", ["Negative", "Neutral", "Positive"])
    physical_activity = st.selectbox("🏃 Physical Activity (Hours)", [0, 1, 2, 3, 4, 5, 6])
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

ordered_columns = [
    'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Sleep_Hours', 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 'Tutoring_Sessions',
    'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Physical_Activity',
    'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

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

# Manual label encoding
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})
data['Internet_Access'] = data['Internet_Access'].map({'Yes': 1, 'No': 0})
data['School_Type'] = data['School_Type'].map({'Public': 1, 'Private': 0})
data['Learning_Disabilities'] = data['Learning_Disabilities'].map({'No': 0, 'Yes': 1})
data['Parental_Involvement'] = data['Parental_Involvement'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Access_to_Resources'] = data['Access_to_Resources'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Motivation_Level'] = data['Motivation_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Family_Income'] = data['Family_Income'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Teacher_Quality'] = data['Teacher_Quality'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Peer_Influence'] = data['Peer_Influence'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
data['Parental_Education_Level'] = data['Parental_Education_Level'].map({'High School': 0, 'College': 1, 'Postgraduate': 2})
data['Distance_from_Home'] = data['Distance_from_Home'].map({'Near': 0, 'Moderate': 1, 'Far': 2})

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
    sns.barplot(x=scaled_data.columns, y=scaled_data.iloc[0], palette='viridis', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Scaled Value (0 to 1)")
    ax.set_title("Impact of Scaled Numerical Features")
    st.pyplot(fig)

    # 📋 Encoded Categorical Features Table
    st.write("### 🧩 Encoded Categorical Features")
    categorical_data = data.drop(columns=numeric_features)
    st.dataframe(categorical_data.T.rename(columns={0: "Encoded Value"}))