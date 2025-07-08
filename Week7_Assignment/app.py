# titanic_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Load trained models and preprocessing pipeline
rf_model = joblib.load('Week7_Assignment/random_forest_model.pkl')
gb_model = joblib.load('Week7_Assignment/gradient_boosting_model.pkl')

# Load the entire preprocessor (ColumnTransformer)
preprocessor = joblib.load('Week7_Assignment/preprocessor.pkl')

# Set page config
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.markdown("""
<style>
.card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction App")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])

model_map = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}
model = model_map[model_choice]

st.sidebar.markdown("### Input Passenger Details")

# User Inputs
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.radio("Sex", ["male", "female"])
        age = st.slider("Age", 1, 80, 30)
        sibsp = st.slider("Siblings/Spouses Aboard", 0, 5, 0)
        parch = st.slider("Parents/Children Aboard", 0, 5, 0)

    with col2:
        fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# DataFrame construction
user_input = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Safe prediction using try-except
try:
    user_transformed = preprocessor.transform(user_input)
    prediction = model.predict(user_transformed)[0]
    pred_proba = model.predict_proba(user_transformed)[0]

    # Output Block
    with st.container():
        st.markdown("### 🎯 Prediction Result")
        if prediction == 1:
            st.success("This passenger is likely to **Survive** 🛟")
        else:
            st.error("This passenger is likely **Not to Survive** ❌")

        # Chart
        chart_df = pd.DataFrame({
            'Outcome': ['Not Survived', 'Survived'],
            'Probability': pred_proba
        })
        bar_chart = alt.Chart(chart_df).mark_bar().encode(
            x='Outcome',
            y='Probability',
            color='Outcome',
            tooltip=['Outcome', 'Probability']
        ).properties(width=500)
        st.altair_chart(bar_chart)

    # Interpretation block
    with st.expander("📘 Model Interpretation Summary"):
        st.write(f"You entered a {age}-year-old {sex} in class {pclass}, who embarked from {embarked}, paid a fare of {fare}, and was traveling with {sibsp} sibling(s)/spouse(s) and {parch} parent(s)/child(ren).")

except Exception as e:
    st.error(f"Something went wrong during prediction: {str(e)}")

# Dataset Summary block
with st.expander("📊 Dataset Statistics"):
    df = pd.read_csv('titanic/train.csv')
    st.write("**Survival Distribution:**")
    st.bar_chart(df['Survived'].value_counts())
    st.write("**Class Distribution:**")
    st.bar_chart(df['Pclass'].value_counts())
