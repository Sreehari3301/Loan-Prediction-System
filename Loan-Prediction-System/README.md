# Loan Prediction & Risk Analysis System

A production-grade Machine Learning CLI application for analyzing lending risk and predicting loan amounts. The system supports dual pipelines:
1. **Regression Task**: Predicts the loan amount (`loan_amnt`) a borrower is likely to obtain, using **Linear Regression**.
2. **Classification Task**: Evaluates creditworthiness and predicts loan default probability (`loan_status` as Fully Paid vs. Default), using **Logistic Regression**.

---

## Table of Contents
- [Features](#-features)
- [Architecture & Design](#-architecture--design)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Model Training](#1-model-training)
  - [2. Model Prediction](#2-model-prediction)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Supported Features Schema](#-supported-features-schema)

---

## Features

- **Dual Task Support**: Train either a regressor (for amount estimation) or classifier (for credit risk evaluation) from the same CLI.
- **Robust Preprocessing Pipeline**: Automatically handles missing values (imputation), scales numeric values, and one-hot encodes categorical values inside a scikit-learn pipeline.
- **Interactive Mode**: Collects customer details dynamically from CLI prompts to generate real-time predictions.
- **Advanced Classification Output**: Displays both predicted status (Fully Paid vs. Default) and exact probability confidence scores.
- **Optimized Data Pipeline**: Handles memory-safe sampling for training on large-scale datasets (e.g., 2M+ rows).

---

## Architecture & Design

The application uses an end-to-end `scikit-learn` Pipeline:
- **Numerical Processing**: Median imputation followed by standard scaling (`StandardScaler`).
- **Categorical Processing**: Most frequent value imputation followed by one-hot encoding (`OneHotEncoder`).
- **Persistence**: Models are saved as compressed serialized files using `joblib` containing the preprocessing pipeline, estimators, and custom metadata (e.g., target task type).

---

## Installation & Setup

1. Navigate to the project directory:
   ```bash
   cd Loan-Prediction-System
   ```

2. Make sure the virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```

3. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure dependencies `pandas`, `numpy`, `scikit-learn`, and `joblib` are installed).*

4. Make sure your dataset is in the `data/` directory (This model is trained by using `Lending Club Loan Data` dataset from Kaggle):
   ```
   data/loan.csv
   ```

---

## Usage Guide

### 1. Model Training

Use the `train` subcommand to build and save a model.

#### Train a Classification Model (Default Risk)
Train a classifier predicting whether a loan will be repaid (`Fully Paid`) or default (`Charged Off`):
```bash
python main.py train --task classification --sample-size 200000 --model model_classifier.joblib
```

#### Train a Regression Model (Loan Amount)
Train a regressor predicting loan amounts:
```bash
python main.py train --task regression --sample-size 200000 --model model_regressor.joblib
```

*Note: `--sample-size` can be set to any integer or `"all"` to train on the entire dataset.*

---

### 2. Model Prediction

Use the `predict` subcommand to generate predictions. The system automatically detects whether the loaded model is a classifier or regressor and adjusts the output accordingly.

#### A. Interactive Mode (Recommended)
Launch a step-by-step interactive command line questionnaire:
```bash
python main.py predict --model model_classifier.joblib --interactive
```

#### B. JSON File or String Input
Pass a JSON file path or a raw JSON string:
```bash
python main.py predict --model model_classifier.joblib --input-json '{"loan_amnt": 15000, "annual_inc": 75000, "dti": 15.4, "revol_bal": 9500, "revol_util": 35.1, "open_acc": 8, "total_acc": 18, "pub_rec": 0, "home_ownership": "MORTGAGE", "purpose": "credit_card", "term": "36 months", "verification_status": "Verified", "application_type": "Individual"}'
```

#### C. Command Line Flags
Provide details directly as CLI flags:
```bash
python main.py predict --model model_classifier.joblib \
  --loan-amnt 12000 \
  --annual-inc 60000 \
  --dti 12.5 \
  --revol-bal 8000 \
  --revol-util 40.0 \
  --open-acc 12 \
  --total-acc 24 \
  --pub-rec 0 \
  --home-ownership RENT \
  --purpose debt_consolidation \
  --term "36 months" \
  --verification-status "Not Verified" \
  --application-type Individual
```

---

## Model Evaluation & Metrics

When training, the system outputs rich performance reports:

### Classification Metrics (Default Risk)
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Success rate when predicting a loan is repaid.
- **Recall**: Percentage of fully repaid loans successfully identified by the model.
- **Confusion Matrix**:
  ```
               Predicted Default    Predicted Paid
  Actual Default         TN                 FP
  Actual Paid            FN                 TP
  ```
- **Classification Report**: Precision, recall, and F1-scores breakdown.

### Regression Metrics (Loan Amount)
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual loan amounts.
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors.
- **R-squared (R2 Score)**: Variance proportion explained by features.

---

## Supported Features Schema

The model requires the following schema for predictions:

| Feature Name | Type | Description / Examples |
| :--- | :--- | :--- |
| `loan_amnt` | Numeric | Requested amount (only required for classification tasks) |
| `annual_inc` | Numeric | Annual income in USD (e.g., `65000`) |
| `dti` | Numeric | Debt-to-income ratio (e.g., `18.5`) |
| `revol_bal` | Numeric | Total revolving credit balance (e.g., `12000`) |
| `revol_util` | Numeric | Revolving line utilization percentage (e.g., `45.2`) |
| `open_acc` | Numeric | Number of open credit lines (e.g., `10`) |
| `total_acc` | Numeric | Total number of credit lines (e.g., `25`) |
| `pub_rec` | Numeric | Number of derogatory public records (e.g., `0`) |
| `home_ownership` | Categorical | `MORTGAGE`, `RENT`, `OWN`, `ANY`, `OTHER` |
| `purpose` | Categorical | `debt_consolidation`, `credit_card`, `home_improvement`, `other`, `major_purchase`, etc. |
| `term` | Categorical | `36 months`, `60 months` |
| `verification_status`| Categorical | `Not Verified`, `Source Verified`, `Verified` |
| `application_type` | Categorical | `Individual`, `Joint App` |
