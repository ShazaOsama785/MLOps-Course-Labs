
# Bank Customer Churn Prediction with MLflow
This project aims to predict whether a bank customer will leave the service (churn) using various machine learning models. It includes full support for data preprocessing, training, evaluation, and model tracking/logging via MLflow.

# Dataset
Source: Churn_Modelling.csv

Key Features: CreditScore, Geography, Gender, Age, Tenure, Balance, etc.

Target: Exited (0 = stayed, 1 = churned)

# Project Workflow
Data Rebalancing
Handles class imbalance using downsampling.

Preprocessing

Scales numerical features

One-hot encodes categorical features (Geography, Gender)

Saves the column transformer to disk

Model Training & Evaluation
Three models are trained and evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

MLflow Logging
Automatically logs:

Parameters

Metrics

Confusion matrix images

Trained model artifacts

Model signature for reproducibility

🤖 Models Used
Model	Accuracy	Notes
Logistic Regression	~70%	Baseline model
Random Forest	~75%	Moved to staging
SVM (RBF kernel)	76%	Deployed to production

🚀 Production Setup
✅ SVM Model (RBF kernel, C=1.0, gamma=scale) gave the best accuracy (76%) and is deployed to Production.

🧪 Random Forest is placed in Staging for further validation.

🧰 Tech Stack
Python

Scikit-learn

Pandas

Matplotlib

MLflow

Pickle

🏁 How to Run
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Start MLflow tracking server locally:

bash
Copy
Edit
mlflow ui
Run the training script:

bash
Copy
Edit
python train.py
📁 Artifacts
column_transformer.pkl: Saved transformer for consistent inference

confusion_matrix_*.png: Visual evaluation

MLflow logs all model artifacts and metrics
=======
# MLOps Course Labs

Welcome to the lab repository for the [MLOps Course](https://github.com/Heba-Atef99/MLOps-Course).

Throughout this hands-on journey, you'll develop a **Bank Customer Churn Prediction** application—starting from the research phase and progressing through the full MLOps lifecycle, all the way to deployment.


