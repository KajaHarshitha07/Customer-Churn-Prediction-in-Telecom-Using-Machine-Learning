
# Streamlit App for Telecom Churn

## How to run
1. Install requirements:
   ```bash
   pip install streamlit scikit-learn imbalanced-learn xgboost joblib pandas numpy
   ```
   *(xgboost is optional; the app will work without it)*

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. In the app:
   - Upload your CSV (e.g., Telco Customer Churn)
   - Set target column (default: Churn)
   - Click **Train pipeline now**
   - Use the **Predict** form to test single cases
   - Download the trained pipeline (`churn_pipeline.joblib`)
