# Predictive Maintenance Model for Industrial Equipment

This project builds a machine learning pipeline to predict equipment failure using operational and sensor-based data. It is designed as a beginner-friendly but credible portfolio project for a data science application or GitHub portfolio.

## Project Goal

The goal of this project is to identify whether a machine is likely to fail based on variables such as temperature, rotational speed, torque, tool wear, vibration, pressure, and machine type. This mirrors real predictive maintenance use cases where organisations want to reduce downtime, improve reliability, and schedule maintenance more effectively.

## Project Structure

```text
predictive_maintenance_project/
│
├── data/
│   └── predictive_maintenance.csv
├── src/
│   └── predictive_maintenance_project.py
├── outputs/
│   └── .gitkeep
├── README.md
├── requirements.txt
└── .gitignore
```

## Tools Used

- Python
- pandas
- scikit-learn
- NumPy
- Logistic Regression
- Random Forest

## What the Project Does

- Loads a predictive maintenance dataset from CSV
- Selects available numeric and categorical features
- Cleans missing values
- Scales numeric variables
- Encodes categorical features
- Trains two classification models
  - Logistic Regression
  - Random Forest
- Evaluates performance using:
  - Confusion Matrix
  - Classification Report
  - ROC AUC
- Shows feature importance using permutation importance

## Example Input Columns

The script is written to work with columns such as:

- `failure`
- `air_temperature`
- `process_temperature`
- `rotational_speed`
- `torque`
- `tool_wear`
- `operating_hours`
- `vibration`
- `pressure`
- `machine_type`

If your dataset uses different column names, update the column lists at the top of the script.

## How to Run the Project

### 1. Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Mac or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place your dataset file in the `data` folder and name it:

```text
predictive_maintenance.csv
```

### 4. Run the script

```bash
python src/predictive_maintenance_project.py
```

## Suggested Dataset Sources

You can use public predictive maintenance datasets from:
- Kaggle
- UCI Machine Learning Repository

Search terms:
- `predictive maintenance dataset`
- `machine failure prediction dataset`
- `industrial sensor failure dataset`

## Suggested Improvements

To make the project stronger over time, you can add:
- XGBoost or LightGBM
- Hyperparameter tuning with GridSearchCV
- Precision-recall analysis for imbalanced classes
- Visualisations of feature distributions
- SHAP feature importance
- A Streamlit dashboard

## CV-Ready Project Description

**Predictive Maintenance Model for Industrial Equipment**

Developed a machine learning model to predict equipment failure using operational and sensor-based data. Built and evaluated Logistic Regression and Random Forest models using Python and scikit-learn, and assessed performance using confusion matrix, classification metrics, and ROC AUC. Analysed feature importance to identify variables most strongly associated with maintenance risk.

## GitHub Portfolio Tip

For a stronger GitHub presentation, add:
- a short `results_summary.md`
- screenshots of outputs
- a notebook version of the project
- a simple chart comparing model performance
