#  Predictive Maintenance using Machine Learning

##  Project Overview
This project develops a machine learning solution to predict equipment failure in advance, enabling proactive maintenance and reducing unplanned downtime in industrial operations.

The solution is deployed as an interactive web application using Streamlit, allowing users to input operational data and receive real-time predictions on equipment failure risk.

---

##  Business Problem
In industrial environments such as manufacturing, energy, and heavy operations, unexpected equipment failure can lead to:

- Significant production downtime  
- High repair and replacement costs  
- Safety risks  
- Reduced operational efficiency  

Traditional maintenance strategies are often:
- **Reactive** (fix after failure), or  
- **Scheduled** (replace regardless of condition)  

This project introduces a **predictive maintenance approach**, enabling data-driven decision-making and early fault detection.

---

##  Objective
To build a machine learning model that:
- Predicts the likelihood of equipment failure  
- Identifies early warning signals  
- Supports proactive and cost-efficient maintenance scheduling  

---

##  Methodology

### 1. Data Preprocessing
- Cleaned dataset and handled missing values  
- Encoded categorical variables  
- Scaled numerical features  

### 2. Exploratory Data Analysis (EDA)
- Identified patterns in operational and sensor data  
- Explored relationships between features and failure events  

### 3. Feature Engineering
- Selected relevant features influencing equipment performance  
- Improved model input quality  

### 4. Model Development
Trained and compared multiple models:
- Logistic Regression  
- Random Forest  

### 5. Model Evaluation
Evaluated models using:
- ROC-AUC score  
- Precision, Recall, and F1-score  
- Confusion matrix  

---

##  Results

- **Logistic Regression AUC:** 0.68  
- **Random Forest AUC:** 0.44  

### Classification Report (Logistic Regression)

- Accuracy: 1.00  
- Precision (Failure Class): 0.00  
- Recall (Failure Class): 0.00  

### ⚠️ Key Observation
The dataset is **highly imbalanced**, with extremely few failure cases compared to non-failure cases.

As a result:
- The model achieved high overall accuracy by predicting the majority class  
- However, it struggled to correctly identify rare failure events  

This highlights a critical challenge in predictive maintenance systems, where failures are rare but highly impactful.

---

##  Key Insights

- Equipment failure is a **rare event**, making prediction inherently challenging  
- Accuracy alone is not a reliable metric for evaluating model performance in imbalanced datasets  
- Models must be optimized for **recall and precision**, especially for failure detection  
- Even moderately predictive models (AUC ~0.68) can provide value when combined with domain knowledge  

---

##  Live Application
 **[Click here to use the deployed Streamlit app](https://predictive-maintenance-mlrn.streamlit.app/)**

Features:
- Input equipment or operational data  
- Receive real-time failure predictions  
- Support maintenance decision-making  

---

##  Business Impact

This solution demonstrates how machine learning can:

- Enable early detection of potential failures  
- Reduce unplanned downtime  
- Improve maintenance scheduling  
- Support transition from reactive to predictive maintenance  

---

##  Limitations & Future Improvements

### Current Limitations
- Severe class imbalance (very few failure instances)  
- Limited ability to detect rare failure events  

### Future Improvements
- Apply resampling techniques (e.g., SMOTE, oversampling)  
- Use cost-sensitive learning or class weighting  
- Explore anomaly detection methods  
- Collect more failure data for improved model performance  
- Experiment with advanced models (e.g., XGBoost, time-series models)  

---

##  Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Streamlit  

---

## 📂 Project Structure
predictive-maintenance-ml/
│
├── data/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
└── README.md


---

##  Author
**Jonathan Makario**  
Mechanical Engineer | Data Science Enthusiast  

Background in industrial operations, energy systems, and project management, with a focus on applying data science to real-world engineering challenges.

---

##  Conclusion
This project demonstrates the application of machine learning to a real-world industrial problem, while also highlighting key challenges such as class imbalance in failure prediction.

It reflects a practical understanding of both the **technical and operational aspects** of predictive maintenance systems.
