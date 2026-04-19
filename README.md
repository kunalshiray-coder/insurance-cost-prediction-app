# Insurance Cost Prediction

An end-to-end data science portfolio project focused on predicting health insurance premium cost using demographic and medical-risk variables. This project covers the complete workflow from exploratory data analysis and hypothesis testing to machine learning model development, deployment, and portfolio presentation.

## Live App
**Streamlit Demo:** https://insurance-cost-prediction-app-lnzzellxzkgdlkj9baapau.streamlit.app/

## GitHub Repository
**Repo:** https://github.com/kunalshiray-coder/insurance-cost-prediction-app

---

## 1. Problem Statement

Insurance companies need to estimate premium costs more accurately using individual-level information rather than relying only on broad actuarial averages. In this project, the goal is to build a machine learning system that predicts **`PremiumPrice`** using health and demographic variables such as age, diabetes, blood pressure problems, transplants, chronic diseases, height, weight, allergies, family cancer history, and surgery history.

This project was selected because it provides a complete end-to-end workflow suitable for a strong portfolio:
- business problem framing
- data analysis
- statistical validation
- machine learning model development
- deployment as a usable web application

---

## 2. Project Objective

The main objective is to predict insurance premium cost accurately and convert the final model into a user-friendly premium prediction app.

### Business goals
- improve pricing precision
- support better risk assessment
- identify major factors driving insurance premium
- demonstrate an end-to-end machine learning workflow
- deploy the final solution for real-time usage

### Target variable
- **Target:** `PremiumPrice`

### Evaluation metrics
The models were evaluated using:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

---

## 3. Dataset Overview

The dataset contains **986 rows** and **11 columns**.

### Features
- `Age`
- `Diabetes`
- `BloodPressureProblems`
- `AnyTransplants`
- `AnyChronicDiseases`
- `Height`
- `Weight`
- `KnownAllergies`
- `HistoryOfCancerInFamily`
- `NumberOfMajorSurgeries`
- `PremiumPrice`

### Data quality summary
- no missing values
- no duplicate rows
- all columns are numeric
- binary health indicators are clean and encoded as 0/1

### Important observation
`PremiumPrice` contains only **24 unique values**, indicating that premium values behave like **pricing slabs / bands** rather than a fully continuous target.

---

## 4. Project Workflow

The project was executed in four major blocks.

### Block 1 — Tableau Visualization
Planned Tableau dashboards include:
- premium distribution
- age-wise premium trends
- count of medical conditions
- premium by surgery history
- BMI / body metric analysis
- summary and business insight dashboard

> Tableau dashboard link: **To be added**

### Block 2 — EDA and Hypothesis Testing

#### Step 1: Data Understanding and Initial Inspection
- loaded and inspected dataset
- checked shape, columns, data types, missing values, duplicates
- identified feature groups and target variable

#### Step 2: Univariate Analysis
- studied distributions of age, height, weight, surgeries, and premium
- analyzed prevalence of binary medical conditions
- checked outliers using IQR method
- identified premium slab behavior

#### Step 3: Bivariate Analysis
- analyzed correlation with `PremiumPrice`
- studied age vs premium, BMI vs premium, surgery count vs premium
- compared premiums across medical-condition groups
- created BMI categories for interpretation

#### Step 4: Hypothesis Testing
- Welch’s t-tests for binary health variables
- ANOVA for surgery groups and BMI categories
- chi-square test for selected categorical association
- regression-based significance check

### Block 3 — Machine Learning Modeling

#### Step 1: Preprocessing and Feature Engineering
Created the following engineered features:
- **BMI**
- **TotalHealthRisks**
- **Age_BMI_Interaction**
- **Age_Surgery_Interaction**

#### Step 2: Baseline Model
- Linear Regression

#### Step 3: Advanced Models
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

#### Step 4: Final Model Interpretation
- model comparison
- feature importance analysis
- residual analysis
- premium slab-wise error analysis

### Block 4 — Deployment
- selected best model
- retrained on full dataset
- serialized model artifacts
- created Streamlit app for real-time premium prediction

---

## 5. Key EDA Findings

### Dataset behavior
- the dataset is clean and ready for modeling with minimal preprocessing
- premium values are slab-based rather than fully continuous
- outlier share is low, so aggressive outlier removal was not necessary

### Most common medical conditions
- **BloodPressureProblems:** 46.86%
- **Diabetes:** 41.99%
- **KnownAllergies:** 21.50%
- **AnyChronicDiseases:** 18.05%
- **HistoryOfCancerInFamily:** 11.76%
- **AnyTransplants:** 5.58%

### Major insights
- **Age** is the strongest driver of premium
- premium rises meaningfully with **major surgery count**
- **AnyTransplants** and **AnyChronicDiseases** create the largest premium jumps
- **BMI** is useful, but weaker than age and major medical severity variables
- interaction features improve predictive power

---

## 6. Hypothesis Testing Summary

The statistical testing phase confirmed that premium differences are not random.

### Statistically significant premium drivers
- `AnyChronicDiseases`
- `AnyTransplants`
- `BloodPressureProblems`
- `Diabetes`
- `HistoryOfCancerInFamily`
- `NumberOfMajorSurgeries`
- `BMI_Category`

### Not statistically significant
- `KnownAllergies`

### Key statistical results
- ANOVA for `NumberOfMajorSurgeries`: significant premium difference across surgery groups
- ANOVA for BMI categories: significant premium difference across BMI categories
- chi-square between `AnyChronicDiseases` and `HistoryOfCancerInFamily`: not significant, indicating both features provide distinct information

---

## 7. Feature Engineering

The following engineered variables were created to improve model performance:

### 1. BMI
\[
BMI = \frac{Weight}{(Height\ in\ meters)^2}
\]

### 2. TotalHealthRisks
Sum of six binary health-risk indicators:
- Diabetes
- BloodPressureProblems
- AnyTransplants
- AnyChronicDiseases
- KnownAllergies
- HistoryOfCancerInFamily

### 3. Age_BMI_Interaction
Captures the combined effect of age and body composition.

### 4. Age_Surgery_Interaction
Captures the combined effect of age and surgery history.

These features were added because insurance pricing is influenced not only by individual variables but also by combined risk patterns.

---

## 8. Model Building and Results

### Baseline Model — Linear Regression

**Training performance**
- MAE: **2642.90**
- RMSE: **3720.45**
- R²: **0.6363**
- MAPE: **11.43%**

**Test performance**
- MAE: **2548.85**
- RMSE: **3453.71**
- R²: **0.7203**
- MAPE: **10.85%**

### Advanced Model Comparison

| Model | Test MAE | Test RMSE | Test R² | Test MAPE |
|---|---:|---:|---:|---:|
| Linear Regression | 2548.85 | 3453.71 | 0.7203 | 10.85% |
| Decision Tree Regressor | 1810.58 | 2901.28 | 0.8026 | 7.24% |
| Random Forest Regressor | **1222.09** | **2171.51** | **0.8894** | **5.04%** |
| Gradient Boosting Regressor | 1522.95 | 2371.07 | 0.8682 | 6.28% |

### Final selected model
**Random Forest Regressor**

Why it was selected:
- lowest test RMSE
- best test R²
- strongest overall balance of performance and stability
- strong feature-importance interpretability

---

## 9. Final Model Performance

### Final test metrics
- **MAE:** **1222.09**
- **RMSE:** **2171.51**
- **R²:** **0.8894**
- **Median Absolute Error:** **534.30**
- **MAPE:** **5.04%**

### Practical prediction accuracy
- within **±1000**: **68.18%**
- within **±2000**: **80.81%**
- within **±3000**: **88.89%**

### Interpretation
The final model performs strongly for a banded premium-pricing problem and is especially reliable for the common premium slabs.

---

## 10. Important Model Insights

Feature importance from the final Random Forest model showed:

- **Age** → strongest driver
- **AnyTransplants**
- **Age_BMI_Interaction**
- **Weight**
- **AnyChronicDiseases**
- **Age_Surgery_Interaction**
- **HistoryOfCancerInFamily**
- **NumberOfMajorSurgeries**
- **BMI**

### Business interpretation
Insurance premium in this dataset is primarily driven by:
1. **Age**
2. **Severe medical history**, especially transplants and chronic diseases
3. **Surgery-related severity**
4. **Combined risk effects**, especially age with BMI and age with surgeries

---

## 11. Deployment

The final model was retrained on the full dataset and deployed through **Streamlit**.

### Deployment workflow
- finalized Random Forest model
- retrained on full data
- saved model as `.joblib`
- saved feature list and metadata as `.json`
- built a Streamlit application to:
  - accept user inputs
  - generate engineered features internally
  - predict premium in real time

### Live app
**Streamlit App:** https://insurance-cost-prediction-app-lnzzellxzkgdlkj9baapau.streamlit.app/

---

## 12. Repository Structure

```text
insurance-cost-prediction-app/
├── app.py
├── requirements.txt
├── .gitignore
├── insurance_premium_random_forest.joblib
├── model_features.json
├── model_metadata.json
└── Insurance_Cost_Prediction.ipynb
```

---

## 13. How to Run the App Locally

### 1. Clone the repository
```bash
git clone https://github.com/kunalshiray-coder/insurance-cost-prediction-app.git
cd insurance-cost-prediction-app
```

### 2. Create / activate environment
Using Conda:
```bash
conda create -n insurance-app python=3.11 -y
conda activate insurance-app
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 14. Recommendations and Future Improvements

### Recommendations
- use the model as a decision-support tool for premium estimation
- monitor performance separately for rare premium slabs
- keep feature engineering logic consistent between training and deployment

### Future improvements
- hyperparameter tuning with Optuna or GridSearchCV
- SHAP-based interpretability for local and global explanations
- deployment logging and monitoring
- better handling of rare premium slabs
- richer UI and production-style API version
- add Tableau dashboard and blog links after publishing

---

## 15. Submission Links

- **GitHub Repository:** https://github.com/kunalshiray-coder/insurance-cost-prediction-app
- **Live App:** https://insurance-cost-prediction-app-lnzzellxzkgdlkj9baapau.streamlit.app/
- **Tableau Public:** To be added
- **Technical Blog:** To be added
- **Loom Demo Video:** To be added
- **Final Submission PDF:** To be added

---

## 16. Author

**Kunal Hiray**  
