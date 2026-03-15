# Predictive Maintenance Model for Industrial Equipment

This project uses machine learning to predict equipment failure from operational and sensor-based data.

## Files
- `analysis_model.ipynb`: notebook for data loading, modelling, and evaluation
- `app.py`: Streamlit app for prediction
- `best_model.pkl`: trained Logistic Regression model
- `scaler.pkl`: saved numerical scaler
- `label_encoder_machine_type.pkl`: saved machine type encoder
- `predictive_maintenance.csv`: synthetic project dataset
- `requirements.txt`: Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
