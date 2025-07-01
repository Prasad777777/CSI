#  Week 5: House Prices - Data Preprocessing & Feature Engineering

##  Objective

This project is part of **Week 5** of the Celebal CSI Data Science Summer Internship. The task was to perform detailed **data preprocessing**, **feature engineering**, and **model evaluation** on a real-world housing dataset in preparation for advanced regression tasks.

---

##  Dataset Used

- **Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Rows**: 1460
- **Columns**: 81
- **Target Variable**: `SalePrice`

---

##  Steps Performed

### 1. Data Exploration
- Inspected dataset shape, types, and content
- Identified numerical and categorical features
- Assessed missing values and potential outliers

### 2. Missing Value Handling
- Features like `PoolQC`, `Fence`, and `Alley` had extensive missing values due to non-existence (not noise)
- Applied domain-based imputations (e.g., filled `GarageYrBlt`, `MasVnrType` logically)

### 3. Target Variable Analysis
- `SalePrice` distribution was right-skewed
- Applied **log transformation** to stabilize variance and reduce skew
- Identified high-value outliers

### 4. Correlation Analysis
- Found strong correlation between `SalePrice` and:
  - `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, etc.
- Visualized relationships using heatmaps

### 5. One-Hot Encoding
- Encoded 34 categorical columns using `pd.get_dummies()`
- Total columns expanded to 239

### 6. Feature Scaling
- Standardized numerical columns using `StandardScaler`
- Helped gradient-based and regularized models perform better

---

##  Model Evaluation

| Model               | RMSE           | R² Score |
|--------------------|----------------|----------|
| Gradient Boosting  | **27808.16**   | **0.899**|
| Ridge Regression   | 30401.85       | 0.880    |
| Random Forest      | 30464.87       | 0.879    |
| Lasso Regression   | 52309.59       | 0.643    |
| Linear Regression  | 52607.91       | 0.639    |

- **Gradient Boosting** performed best due to its ensemble strength.
- **Ridge Regression** followed closely.
- **Lasso and Linear Regression** struggled with high-dimensional, sparse data.

---

##  Learnings

- Cleaned and preprocessed a complex real-world dataset
- Learned practical feature engineering techniques
- Understood the significance of target distribution normalization
- Evaluated performance of multiple regression algorithms

---

##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

---

## Files in this Folder

- `week5_house_prices.ipynb`: Main notebook
- `README.md`: Project documentation
- `house-prices-advanced-regression-techniques/`: Raw  dataset files

