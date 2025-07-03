import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
import base64

# Page settings
st.set_page_config(page_title="Student Performance Predictor", layout="wide", page_icon="📊")

# Load model
working_dir = os.path.dirname(os.path.abspath(__file__))
performance_model = pickle.load(open(f"{working_dir}/student_performance_model.sav", 'rb'))

# Title and credits
st.title("🎓 Student Performance Predictor")
st.markdown("""
Developed by **Prasad Baban Parjane**  
🔗 [GitHub Repository](https://github.com/Prasad777777/CSI/tree/main/Final_Project)
""")

st.write("Fill in the details below to predict the student's performance score.")

# Input layout
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

# Create DataFrame
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

# Reorder columns
ordered_columns = list(data.columns)
data = data[ordered_columns]

# Scaling function
def custom_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Apply scaling
data['Attendance'] = custom_scale(data['Attendance'][0], 60, 100)
data['Hours_Studied'] = custom_scale(data['Hours_Studied'][0], 1, 44)
data['Previous_Scores'] = custom_scale(data['Previous_Scores'][0], 50, 100)
data['Sleep_Hours'] = custom_scale(data['Sleep_Hours'][0], 4, 10)
data['Tutoring_Sessions'] = custom_scale(data['Tutoring_Sessions'][0], 0, 8)
data['Physical_Activity'] = custom_scale(data['Physical_Activity'][0], 0, 6)

# Encoding
encode_map = {
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
for col, mapping in encode_map.items():
    data[col] = data[col].map(mapping)

# Prediction
st.write("### 🔍 Prediction Result")
if st.button("🚀 Predict"):
    prediction = performance_model.predict(data)
    st.success(f"🎯 Predicted Score: **{prediction[0]:.2f}**")

    # --- Download PDF Report ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="🎓 Student Performance Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    for col in data.columns:
        val = data[col].values[0]
        pdf.cell(0, 10, f"{col}: {val}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 10, f"🎯 Predicted Score: {prediction[0]:.2f}", ln=True)

    report_path = os.path.join(working_dir, "student_report.pdf")
    pdf.output(report_path)

    with open(report_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="student_report.pdf">📥 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    # --- Visualizations ---
    st.write("### 📊 Visualizations")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=data.columns, y=data.iloc[0].values, palette="viridis", ax=ax)
    ax.set_title("Input Feature Values")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)
