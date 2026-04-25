# From EDA to Deployment: Building an Insurance Cost Prediction System with Machine Learning

## Introduction

Insurance pricing is one of the most practical applications of predictive analytics. If premiums are set too low, insurers absorb unnecessary risk. If they are set too high, good customers may churn and the pricing process may feel unfair or opaque. That tension makes premium prediction a strong machine learning problem: it is measurable, financially relevant, and easy to translate into a real product.

In this project, I built an end-to-end **Insurance Cost Prediction** system that estimates an individual’s insurance premium using demographic and health-related information. The project followed a complete workflow: exploratory data analysis, hypothesis testing, feature engineering, regression modeling, Tableau storytelling, and deployment of the final model as a Streamlit web application.

This was not just a modeling exercise. The project was designed to answer three practical questions:

1. Which factors most strongly influence premium pricing?
2. How accurately can machine learning predict premium values?
3. Can the final model be packaged into a simple product that a non-technical user can actually use?

The finished project includes:
- a cleaned and analyzed Jupyter notebook,
- a Tableau Public story with two dashboards,
- a deployed Streamlit application,
- and a GitHub repository containing the end-to-end workflow.

**Project assets**
- **GitHub Repository:** https://github.com/kunalshiray-coder/insurance-cost-prediction-app
- **Deployed Streamlit App:** https://insurance-cost-prediction-app-lnzzellxzkgdlkj9baapau.streamlit.app/
- **Tableau Public Story:** https://public.tableau.com/views/InsuranceCostPrediction_17771013116120/InsuranceCostPredictionStory?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

---

## Business problem

Insurance companies need to estimate premium costs more accurately using individual-level information rather than relying only on broad actuarial averages or generic pricing rules. In this project, the goal is to build a machine learning system that predicts **PremiumPrice** using health and demographic variables such as age, diabetes, blood pressure problems, transplants, chronic diseases, height, weight, allergies, family cancer history, and surgery history.

From a business perspective, solving this problem creates value in several ways:
- better premium alignment with actual risk,
- more competitive pricing,
- improved customer trust through consistency,
- and the ability to turn the model into a practical premium calculator.

This is also a strong portfolio project because it goes beyond training a model. It demonstrates complete workflow maturity: business framing, data analysis, statistical validation, machine learning, dashboarding, and deployment.

---

## Dataset overview

The dataset contains **986 rows** and **11 columns**.

### Features
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
- PremiumPrice

### Data quality summary
A quick inspection showed that the dataset was clean and easy to work with:
- **0 missing values**
- **0 duplicate rows**
- all fields numeric
- binary medical indicators already encoded as 0/1

That allowed the project to focus more on insight generation, feature engineering, modeling, and deployment rather than heavy cleaning work.

One important caveat emerged early: the project brief mentions smoker-based hypothesis testing as an example, but the actual dataset does **not** contain a smoker variable. So all analyses were restricted to the fields that actually existed in the data.

---

## Exploratory Data Analysis

The first phase focused on understanding the structure of the data and how insurance premium behaves across different subgroups.

### Univariate findings

The numeric variables were generally stable and well behaved:

- **Age** ranged from 18 to 66 with mean **41.75** and median **42**
- **Height** ranged from 145 cm to 188 cm with mean **168.18 cm**
- **Weight** ranged from 51 kg to 132 kg with mean **76.95 kg**
- **NumberOfMajorSurgeries** ranged from 0 to 3 with mean **0.67**
- **PremiumPrice** ranged from **15,000 to 40,000**, with mean **24,336.71** and median **23,000**

Outlier analysis using the IQR rule showed that the dataset was relatively stable:
- Age: 0 outliers
- Height: 0 outliers
- Weight: 16 outliers (1.62%)
- NumberOfMajorSurgeries: 16 outliers (1.62%)
- PremiumPrice: 6 outliers (0.61%)

This meant aggressive outlier treatment was unnecessary.

### Premium is banded, not fully continuous

One of the strongest early insights came from the target variable itself. `PremiumPrice` had only **24 unique values**, meaning the target behaves more like a **slab-based pricing system** than a fully continuous variable.

The most common premium values were:
- **23,000**: 249 records (25.25%)
- **15,000**: 202 records (20.49%)
- **28,000**: 132 records (13.39%)
- **25,000**: 103 records (10.45%)

This shaped the rest of the project because it explained why a linear model could perform reasonably well but still struggle to capture exact jumps between premium bands.

### Health-condition prevalence

Among the binary condition variables, the most common were:
- **BloodPressureProblems:** 46.86%
- **Diabetes:** 41.99%
- **KnownAllergies:** 21.50%
- **AnyChronicDiseases:** 18.05%
- **HistoryOfCancerInFamily:** 11.76%
- **AnyTransplants:** 5.58%

This immediately suggested that blood pressure problems and diabetes could matter because of prevalence, while transplant history could matter because of severity despite being rare.

---

## Bivariate analysis: what actually shifts premium?

The next stage focused on relationships with `PremiumPrice`.

### Age is the strongest premium driver

Age had the strongest linear relationship with premium, with correlation around **0.70**. The average premium rose steadily across age groups:

- **18–25:** ~16,212
- **26–35:** ~21,390
- **36–45:** ~25,486
- **46–55:** ~28,122
- **56–66:** ~28,788

This was the clearest pattern in the dataset and remained important throughout the project.

### Surgery history matters

Premium increased meaningfully with the number of major surgeries:

- **0 surgeries:** ~22,969
- **1 surgery:** ~24,742
- **2 surgeries:** ~28,084
- **3 surgeries:** ~28,000

The distribution is concentrated at 0 and 1 surgery, so higher categories are based on fewer observations, but the upward shift was still strong.

### Chronic disease and transplant history create major premium jumps

Comparing group means showed the biggest premium increases for:
- **AnyTransplants:** +7,865.68
- **AnyChronicDiseases:** +3,387.11
- **BloodPressureProblems:** +2,091.18
- **HistoryOfCancerInFamily:** +1,611.49
- **Diabetes:** +964.32
- **KnownAllergies:** +184.00

This showed that medical severity variables matter much more than milder indicators like allergies.

### BMI is useful, but weaker than age and medical severity

BMI and weight had only weak direct correlations with premium:
- **Weight:** ~0.14
- **BMI:** ~0.10
- **Height:** ~0.03

Still, BMI category analysis showed a gradual increase in average premium from underweight to obese groups. So BMI remained useful as a derived feature even though it was not the dominant driver on its own.

---

## Hypothesis testing

To move from visual patterns to statistical validation, I ran formal tests on the main candidate drivers.

### Welch’s t-tests for binary health variables

The following variables showed statistically significant premium differences at the 5% level:
- **AnyChronicDiseases:** p ≈ 1.73e-13
- **AnyTransplants:** p ≈ 5.54e-08
- **BloodPressureProblems:** p ≈ 9.81e-08
- **Diabetes:** p ≈ 0.0145
- **HistoryOfCancerInFamily:** p ≈ 0.0198

Only **KnownAllergies** was not significant:
- **KnownAllergies:** p ≈ 0.7043

### ANOVA for surgeries and BMI category

- **NumberOfMajorSurgeries** vs premium: p ≈ **2.87e-16**
- **BMI category** vs premium: p ≈ **0.0065**

These results confirmed that surgery history is a strong premium differentiator, while BMI is significant but weaker.

### Multivariate regression check

An OLS regression-based significance check showed that the following variables remained important after controlling for others:
- Age
- BMI
- AnyTransplants
- AnyChronicDiseases
- HistoryOfCancerInFamily
- NumberOfMajorSurgeries

The OLS model achieved **R² ≈ 0.636**, which supported the idea that premium pricing is structured and learnable even with a relatively simple linear form.

---

## Feature engineering

To improve predictive power and reflect business logic better, I created four engineered features:

1. **BMI** = Weight / (Height in meters)^2  
2. **TotalHealthRisks** = sum of the six binary risk indicators  
3. **Age_BMI_Interaction** = Age × BMI  
4. **Age_Surgery_Interaction** = Age × NumberOfMajorSurgeries  

These were designed to capture cumulative burden and interaction effects.

The strongest engineered-feature relationship with premium was:
- **Age_BMI_Interaction:** correlation with premium ≈ **0.664**

This mattered because BMI alone was weak, but BMI combined with age became much more informative.

---

## Model building and comparison

The modeling stage started with a baseline and then moved to more flexible tree-based models.

### Baseline: Linear Regression

**Training performance**
- **MAE:** 2642.90
- **RMSE:** 3720.45
- **R²:** 0.6363
- **MAPE:** 11.43%

**Test performance**
- **MAE:** 2548.85
- **RMSE:** 3453.71
- **R²:** 0.7203
- **MAPE:** 10.85%

This was already a strong baseline and confirmed that the feature set had real predictive signal.

### Advanced models

#### Decision Tree Regressor
- **Test MAE:** 1810.58
- **Test RMSE:** 2901.28
- **Test R²:** 0.8026
- **Test MAPE:** 7.24%

#### Random Forest Regressor
- **Test MAE:** 1222.09
- **Test RMSE:** 2171.51
- **Test R²:** 0.8894
- **Test MAPE:** 5.04%

#### Gradient Boosting Regressor
- **Test MAE:** 1522.95
- **Test RMSE:** 2371.07
- **Test R²:** 0.8682
- **Test MAPE:** 6.28%

### Best model: Random Forest

Random Forest was clearly the strongest performer. Relative to the linear baseline, it improved:
- **Test RMSE** by about **37.13%**
- **Test MAE** by about **52.05%**
- **R²** from **0.7203** to **0.8894**
- **MAPE** from **10.85%** to **5.04%**

This strongly suggests that premium pricing in this dataset is driven by non-linear rules and interactions rather than only straight-line effects.

---

## Final model interpretation

Feature importance from the final Random Forest model aligned closely with the EDA and statistical findings:

- **Age:** 0.6545
- **AnyTransplants:** 0.0953
- **Age_BMI_Interaction:** 0.0654
- **Weight:** 0.0494
- **AnyChronicDiseases:** 0.0387

This was one of the most satisfying parts of the project because the model’s internal ranking agreed with the business logic emerging from the earlier steps.

### Error analysis

The model was also practically useful:
- within **±1000** for **68.18%** of test cases
- within **±2000** for **80.81%** of test cases
- within **±3000** for **88.89%** of test cases

It performed especially well on common premium slabs and was slightly less accurate on rarer higher-end slabs, which is expected because the pricing bands are not equally represented.

---

## Tableau storytelling

To complement the notebook analysis, I created a Tableau Public story with two dashboards.

### 1. Executive Overview
This dashboard summarizes:
- average premium,
- average age,
- average BMI,
- premium distribution,
- and health-condition prevalence.

The premium distribution chart clearly shows that pricing is concentrated in a limited number of slabs, while the health-condition count chart shows that blood pressure problems and diabetes are the most prevalent conditions.

### 2. Premium Drivers
This dashboard explains what pushes premium upward through:
- average premium by age group,
- premium distribution by number of major surgeries,
- chronic disease and transplant comparisons,
- BMI vs premium scatter,
- and average premium by total health risks.

Together, these dashboards translate the notebook findings into an executive-friendly format and make the project easier to communicate to non-technical reviewers.

**Tableau Public Story**
https://public.tableau.com/views/InsuranceCostPrediction_17771013116120/InsuranceCostPredictionStory?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

---

## Deployment with Streamlit

The final phase of the project was deployment.

I packaged the selected Random Forest model and the same feature engineering logic into a Streamlit application. The app accepts raw user inputs such as:
- age,
- height,
- weight,
- diabetes status,
- blood pressure problems,
- chronic disease,
- transplant history,
- family cancer history,
- known allergies,
- and number of major surgeries.

The app then automatically generates:
- BMI,
- TotalHealthRisks,
- Age_BMI_Interaction,
- Age_Surgery_Interaction,

and returns an estimated premium.

I also added extra user-facing outputs such as:
- BMI value,
- BMI category,
- total health-risk count,
- and a simple risk-profile label.

This step was important because it turned the notebook into an actual product. Instead of showing only code and metrics, the project now includes a usable application that a recruiter or stakeholder can interact with directly.

**Live App**
https://insurance-cost-prediction-app-lnzzellxzkgdlkj9baapau.streamlit.app/

---

## Key recommendations

Based on the full project workflow, the main recommendations are:

1. **Age should remain the primary premium driver** because it consistently explains the largest share of variation.
2. **Severe medical-history variables should be weighted carefully**, especially transplant history, chronic disease, and surgery burden.
3. **Mild indicators like allergies should not be overemphasized**, because they were not statistically meaningful in this dataset.
4. **Interaction features materially improve performance**, especially when they combine age with BMI or surgery history.
5. **Tree-based ensemble models are better suited than linear models** for slab-like pricing systems with non-linear structure.

---

## Final conclusion

This project demonstrates what an end-to-end applied data science workflow looks like in practice.

Starting from a clean tabular dataset, I:
- explored the structure of premium pricing,
- validated key drivers statistically,
- engineered more meaningful features,
- compared multiple regression models,
- selected a high-performing Random Forest model,
- built Tableau dashboards for storytelling,
- and deployed the final solution as a Streamlit premium calculator.

The final model achieved **R² = 0.8894** on the test set with **MAPE = 5.04%**, making it both accurate and practical for demonstration.

For a portfolio project, the biggest strength is not just the final metric. It is the completeness of the workflow: it explains **what** the premium should be, **why** the model believes that, and **how** the result can be used in a real interface.
