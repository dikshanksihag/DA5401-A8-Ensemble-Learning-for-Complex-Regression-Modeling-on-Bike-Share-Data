# DA5401-A8-Ensemble-Learning-for-Complex-Regression-Modeling-on-Bike-Share-Data
## Author: Dikshank Sihag (DA25M009)

# 🚲 Ensemble Learning for Complex Regression Modeling on Bike Share Data

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📘 Executive Summary

This project investigates **ensemble regression techniques** to predict **bike rental demand** from environmental and temporal features using the **UCI Bike Sharing Dataset**.  
The study evaluates how **bagging**, **boosting**, and **stacking** improve model accuracy and robustness compared to baseline regressors.

Models compared:

- Model A: Linear Regression (Baseline)  
- Model B: Decision Tree Regressor  
- Model C: Random Forest (Bagging)  
- Model D: Gradient Boosting Regressor  
- Model E: Stacking Regressor (meta-model ensemble)

Performance was measured with **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.

Results show ensemble models significantly outperform single estimators, with **Stacking and Gradient Boosting** achieving the highest predictive power. Bagging (Random Forest) also improved generalization and reduced overfitting.

---

## 🧠 Theoretical Background

### Ensemble Learning Overview
Ensemble learning combines multiple models to improve prediction accuracy and stability. The two major strategies used:

| Strategy | Concept | Examples | Benefit |
|:----------|:---------|:----------|:----------|
| Bagging | Parallel aggregation of weak learners | Random Forest | Reduces variance |
| Boosting | Sequential error correction | Gradient Boosting, XGBoost | Reduces bias |
| Stacking | Layered meta-model combination | Linear meta-learner on RF & GB | Blends strengths |

Ensemble learning leverages diversity among base models to achieve more stable predictions — especially valuable for **complex, noisy, and non-linear data** like bike rentals.

---

## ⚙️ Implementation Overview

### Part A – Data Preprocessing

- Dataset: UCI Bike Share Data  
- Features: `temp`, `humidity`, `windspeed`, `season`, `holiday`, `workingday`, `weather`, `datetime` (converted to hour/day features)  
- Target: `count` (number of bikes rented)  
- Encoding: One-Hot Encoding for categorical variables  
- Scaling: StandardScaler for numerical predictors  
- Split: 80% train / 20% test

---

### Part B – Model Training

| Model | Algorithm | Type | Key Parameters |
|:------|:-----------|:------|:---------------|
| A | Linear Regression | Baseline | None |
| B | Decision Tree | Single learner | max_depth=6 |
| C | Random Forest | Bagging | n_estimators=50 |
| D | Gradient Boosting | Boosting | n_estimators=200, learning_rate=0.1 |
| E | Stacking Regressor | Meta Ensemble | estimators=[KNN, GBR, Bag], meta=LR |

---

### Part C – Evaluation Metrics
- **RMSE** – penalizes larger errors, emphasizes robustness

---

### Part D – Results Summary

| Model                           | Description / Composition                                          | RMSE       | Variance   |
|---------------------------------|--------------------------------------------------------------------|------------:|------------:|
| **Decision Tree (Baseline)**    | Single Decision Tree (max_depth=6)                                 | 118.456     | 14027.081   |
| **Linear Regression (Baseline)**| Ordinary Least Squares Linear Regression                           | 100.446     | 10071.428   |
| **Bagging Regressor**           | Ensemble of Decision Trees (Bootstrap Aggregation)                 | 112.350     | 12617.702   |
| **Gradient Boosting Regressor** | Sequential Ensemble (Bias Reduction via Residual Learning)         | 64.482      | 4156.360    |
| **Stacking Regressor**          | Meta-Ensemble (KNN + Bagging + GB → Ridge Meta-Learner)            | 61.763      | 3813.945    |

<img width="824" height="590" alt="image" src="https://github.com/user-attachments/assets/52e6a680-ef34-45a2-ab9b-2699e2338daf" />



> Ensemble learners (C–E) outperform baselines, with **Stacking** offering best balance of bias–variance and generalization.

---

## 💡 Insights & Discussion

### Bagging vs Boosting
- **Bagging (Random Forest)** reduces overfitting by averaging diverse trees.  
- **Boosting (GBR)** corrects errors iteratively, achieving lower bias but can overfit small data.  
- In this dataset, Gradient Boosting outperformed RF slightly due to its sequential optimization.

### Stacking Advantages
Stacking combines the diversity of Random Forest and Gradient Boosting, improving predictions through meta-learning.  
This ensemble demonstrated the lowest RMSE.

### Practical Implication
For real-world demand forecasting where both **trend** and **random effects** exist, stacking provides resilience against temporal and weather variability.

---

## 🧰 Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/<your-username>/BikeShare-Ensemble-Regression.git
cd BikeShare-Ensemble-Regression
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook Assignment_8_DAL.ipynb
```

---

## 📊 Conclusion

Ensemble models outperform traditional regressors in predicting bike rental demand.  
- Random Forest and Gradient Boosting provide robust individual performance.  
- Stacking yields optimal accuracy, combining both bias reduction and variance control.

These results highlight the power of ensemble learning in handling **complex, non-linear real-world regression problems**.

---

**© 2025 Dikshank Sihag | DA5401 – Advanced Data Analytics**
