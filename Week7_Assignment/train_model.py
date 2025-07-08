# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Load Titanic dataset
df = pd.read_csv('train.csv')

# Keep relevant features and drop rows with missing values
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].dropna()

# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Preprocessing pipeline
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

# Transform features
X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Save models and preprocessor
joblib.dump(rf_model, 'Week7_Assignment/random_forest_model.pkl')
joblib.dump(gb_model, 'Week7_Assignment/gradient_boosting_model.pkl')
joblib.dump(preprocessor, 'Week7_Assignment/preprocessor.pkl')

print("✅ Models and preprocessor saved successfully.")
