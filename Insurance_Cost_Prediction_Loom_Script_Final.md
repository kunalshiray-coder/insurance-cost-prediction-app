# 5-Minute Loom Video Script — Insurance Cost Prediction
**Presenter: Kunal Hiray**

---

## Recommended screen flow

1. GitHub README / project title
2. Notebook introduction and dataset overview
3. Tableau Story — Executive Overview
4. Tableau Story — Premium Drivers
5. Notebook model comparison section
6. Feature importance / final model results
7. Streamlit app demo
8. Final wrap-up with links

---

## Full speaking script

### 0:00–0:30 — Introduction

**What to show on screen**
- GitHub repository or project README
- Project title: **Insurance Cost Prediction**

**What to say**
Hello everyone, my name is **Kunal Hiray**, and in this video I’m presenting my portfolio project on **Insurance Cost Prediction**.

The objective of this project is to predict health insurance premium price using customer demographic and medical-risk information. In this project, I covered the complete data science workflow including **EDA, hypothesis testing, machine learning modeling, Tableau visualization, and deployment using Streamlit**.

---

### 0:30–1:00 — Problem Statement and Business Goal

**What to show**
- README problem statement section or notebook introduction

**What to say**
This project solves a very practical business problem for insurance companies. The goal is to estimate premium prices more accurately instead of relying only on broad actuarial averages or generic rule-based pricing.

Better premium prediction helps insurers improve pricing precision, strengthen risk assessment, remain competitive, and offer more consistent pricing to customers.

The target variable in this project is **PremiumPrice**, so this is a **supervised regression problem**.

---

### 1:00–1:40 — Dataset Overview and Initial Findings

**What to show**
- Dataset shape and columns from notebook
- Initial inspection results

**What to say**
The dataset contains **986 rows and 11 columns**. The main variables are:
- Age
- Diabetes
- BloodPressureProblems
- AnyTransplants
- AnyChronicDiseases
- Height
- Weight
- KnownAllergies
- HistoryOfCancerInFamily
- NumberOfMajorSurgeries
- and PremiumPrice

During initial inspection, I found that the dataset was clean:
- **no missing values**
- **no duplicate rows**
- and all variables were already numeric.

One important early observation was that **PremiumPrice had only 24 unique values**, which means the premium behaves more like a **slab-based pricing system** than a fully continuous variable.

---

### 1:40–2:20 — EDA and Tableau Findings

**What to show**
- Tableau Story — Executive Overview
- Then Tableau Story — Premium Drivers

**What to say**
In the exploratory analysis and Tableau dashboards, a few strong patterns emerged.

First, **age turned out to be the strongest premium driver**. Average premium increased steadily across age groups.

Second, among the medical conditions, the most common were:
- **BloodPressureProblems**
- and **Diabetes**

Third, more severe medical-history variables had much stronger premium impact. For example:
- **AnyTransplants**
- **AnyChronicDiseases**
- and **NumberOfMajorSurgeries**

The Tableau story helped summarize this in two views:
- an **Executive Overview** dashboard for premium distribution and condition prevalence,
- and a **Premium Drivers** dashboard showing age, surgeries, chronic disease, transplant status, BMI, and total health-risk effects.

---

### 2:20–2:55 — Hypothesis Testing

**What to show**
- notebook hypothesis-testing section
- or selected result tables

**What to say**
After EDA, I used hypothesis testing to validate which features were statistically significant.

The key findings were:
- **AnyChronicDiseases** showed a highly significant premium difference
- **AnyTransplants** also had a very strong premium impact
- **BloodPressureProblems**, **Diabetes**, and **HistoryOfCancerInFamily** were statistically significant
- **KnownAllergies** was not statistically significant

I also ran ANOVA tests, which confirmed that:
- premium differed significantly across **surgery groups**
- and also across **BMI categories**

This helped validate that premium pricing in this dataset is influenced more by **medical severity and long-term health risk** than by weaker variables like allergies.

---

### 2:55–3:40 — Feature Engineering and Model Building

**What to show**
- Feature engineering section
- Model comparison table

**What to say**
In the modeling phase, I created four engineered features:
- **BMI**
- **TotalHealthRisks**
- **Age_BMI_Interaction**
- **Age_Surgery_Interaction**

These were added because premium pricing is influenced not only by individual variables, but also by combined risk patterns.

I started with **Linear Regression** as a baseline model, and then compared it with:
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

The best model was **Random Forest Regressor**.

Its final test performance was:
- **MAE: 1222.09**
- **RMSE: 2171.51**
- **R²: 0.8894**
- **MAPE: 5.04%**

This clearly outperformed the linear baseline and confirmed that the premium prediction problem contains strong **non-linear relationships**.

---

### 3:40–4:10 — Final Model Interpretation

**What to show**
- Feature importance chart
- final model results section

**What to say**
Feature importance from the Random Forest model showed that the strongest drivers were:
- **Age**
- **AnyTransplants**
- **Age_BMI_Interaction**
- **Weight**
- **AnyChronicDiseases**

This was consistent with the earlier EDA and hypothesis testing.

So the main conclusion is that premium pricing in this dataset is driven primarily by:
- **Age**
- **major medical severity**
- **chronic disease burden**
- and **interaction effects between health and demographic variables**

---

### 4:10–4:45 — Deployment Demo

**What to show**
- Open the Streamlit app
- enter a sample profile
- click predict

**What to say**
After selecting the final model, I deployed it as a **Streamlit web application**.

In this app, a user can enter profile details like:
- age
- height
- weight
- diabetes
- blood pressure problems
- chronic disease
- transplant history
- family cancer history
- allergies
- and number of major surgeries

The app automatically creates derived features like:
- **BMI**
- **TotalHealthRisks**
- **Age_BMI_Interaction**
- **Age_Surgery_Interaction**

and then returns:
- the **estimated premium**
- BMI and BMI category
- total health-risk count
- and a simple risk-profile summary

This makes the project much more practical because it demonstrates not only modeling, but also deployment and usability.

---

### 4:45–5:10 — Conclusion

**What to show**
- GitHub repo
- Tableau link
- Streamlit app link
- README

**What to say**
To conclude, this project demonstrates the full end-to-end data science lifecycle:
- business understanding
- EDA
- hypothesis testing
- feature engineering
- model comparison
- dashboard storytelling
- and deployment

The final Random Forest model achieved **R² = 0.8894** with **MAPE = 5.04%**, and the deployed app makes the solution easy to demonstrate and use.

Thank you for watching.  
I’m **Kunal Hiray**, and this was my Insurance Cost Prediction portfolio project.

---

## Short backup version for natural delivery

Hi, I’m **Kunal Hiray**, and this is my Insurance Cost Prediction project.

The aim of this project is to predict insurance premium more accurately using demographic and health-related features. The dataset contains 986 rows and 11 columns, and it was clean with no missing values.

During EDA, I found that premium behaves in pricing slabs, age is the strongest premium driver, and medical severity variables such as chronic disease, transplant history, and surgeries have much stronger effects than mild variables like allergies.

I validated these findings using statistical tests, then built machine learning models including Linear Regression, Decision Tree, Random Forest, and Gradient Boosting. The best model was Random Forest, which achieved an R² of 0.8894 and a MAPE of 5.04 percent on the test set.

To make the project practical, I also deployed the final model as a Streamlit app where users can enter their profile and get a premium estimate instantly.

Overall, this project demonstrates the full workflow from analysis to modeling to dashboarding and deployment.

Thank you for watching.
