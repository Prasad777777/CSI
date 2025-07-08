# titanic_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load trained models
rf_model = joblib.load('Week7_Assignment/random_forest_model.pkl')
gb_model = joblib.load('Week7_Assignment/gradient_boosting_model.pkl')

# Sample training data (10 rows only for fitting preprocessor & summary)
sample_data = StringIO("""
Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
1,1,female,29,0,0,211.34,S
0,3,male,35,0,0,8.05,S
1,2,female,26,1,0,26.00,S
0,3,male,30,0,0,7.23,C
0,1,male,54,0,0,51.86,S
1,3,female,2,3,1,21.08,S
0,3,male,20,0,0,7.23,Q
1,1,female,38,1,0,71.28,C
0,3,male,23,0,0,7.85,S
1,2,female,30,0,0,10.50,S
""")
df_train = pd.read_csv(sample_data)

# Define preprocessing pipeline
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
numeric_transformer = StandardScaler()

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit preprocessor on sample data
preprocessor.fit(df_train.drop(columns=['Survived']))

# Streamlit UI setup
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

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])
model = {"Random Forest": rf_model, "Gradient Boosting": gb_model}[model_choice]

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

# Construct input DataFrame
user_input = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Prediction
try:
    user_transformed = preprocessor.transform(user_input)
    prediction = model.predict(user_transformed)[0]
    pred_proba = model.predict_proba(user_transformed)[0]

    # Prediction Output
    with st.container():
        st.markdown("### 🎯 Prediction Result")
        if prediction == 1:
            st.success("This passenger is likely to **Survive** 🛟")
        else:
            st.error("This passenger is likely **Not to Survive** ❌")

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

    with st.expander("📘 Model Interpretation Summary"):
        st.write(f"You entered a {age}-year-old {sex} in class {pclass}, who embarked from {embarked}, paid a fare of {fare}, and was traveling with {sibsp} sibling(s)/spouse(s) and {parch} parent(s)/child(ren).")

except Exception as e:
    st.error(f"Something went wrong during prediction: {str(e)}")

# Dataset Summary block using same embedded dataset
with st.expander("📊 Dataset Statistics"):
    st.write("**Survival Distribution:**")
    st.bar_chart(df_train['Survived'].value_counts())
    st.write("**Class Distribution:**")
    st.bar_chart(df_train['Pclass'].value_counts())
