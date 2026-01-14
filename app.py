#!pip install streamlit scikit-learn imbalanced-learn xgboost joblib pandas numpy
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(page_title="Telecom Churn App", layout="wide")
st.title("📊 Telecom Customer Churn — Interactive App")

st.markdown("""
This app can **load an existing trained pipeline** (`churn_pipeline.joblib`) or **train one from a CSV**.
It follows a typical workflow used in many churn projects:
- Drop ID-like columns (e.g., `customerID`)
- Coerce `TotalCharges` to numeric
- Map binary Yes/No columns to 1/0
- One-Hot Encode remaining categoricals; StandardScale numerics
- Optional **SMOTE** to handle class imbalance
""")

@st.cache_data
def read_csv(file):
    return pd.read_csv(file)

def clean_dataframe(df: pd.DataFrame):
    # Drop ID columns
    id_cols = [c for c in df.columns if c.lower().replace("_","").replace(" ","") in ["customerid","customer_id"]]
    if id_cols:
        df = df.drop(columns=id_cols)
    # Coerce TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Strip and unify Yes/No style
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            if set(x.lower() for x in df[c].unique()) <= {"yes","no"}:
                df[c] = df[c].str.lower().map({"yes":1,"no":0})
    # Basic missing
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df

def build_preprocessor(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]) if target in df.columns else df.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    return pre, num_cols, cat_cols

def get_model(model_name: str):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "SGD": SGDClassifier(random_state=42),
        "GaussianNB": GaussianNB()
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
    return models[model_name]

def train_pipeline(df: pd.DataFrame, target: str, model_name: str, use_smote: bool = True):
    pre, num_cols, cat_cols = build_preprocessor(df, target)
    model = get_model(model_name)
    if use_smote:
        pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("clf", model)])
    else:
        pipe = Pipeline([("pre", pre), ("clf", model)])
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1], zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    report = classification_report(y_test, y_pred, digits=3)
    metrics = {
        "accuracy": acc,
        "precision_0": pr[0], "precision_1": pr[1],
        "recall_0": rc[0], "recall_1": rc[1],
        "f1_0": f1[0], "f1_1": f1[1],
        "confusion_matrix": cm.tolist(),
        "report": report
    }
    return pipe, metrics, X

def make_prediction_form(df_like: pd.DataFrame, target: str):
    X_cols = [c for c in df_like.columns if c != target]
    st.subheader("🧮 Single Prediction Form")
    inputs = {}
    left, right = st.columns(2)
    for i, c in enumerate(X_cols):
        col = left if i % 2 == 0 else right
        with col:
            if df_like[c].dtype.kind in "biufc":
                default = float(df_like[c].median())
                val = st.number_input(c, value=float(default))
                inputs[c] = val
            else:
                opts = sorted([str(x) for x in df_like[c].dropna().unique()])[:100]
                if len(opts) == 0:
                    opts = [""]
                sel = st.selectbox(c, opts)
                inputs[c] = sel
    return pd.DataFrame([inputs])

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    load_existing = st.toggle("Load existing pipeline (churn_pipeline.joblib)", value=False)
    model_name = st.selectbox("Model", ["Random Forest","Logistic Regression","SVM","KNN","Decision Tree","Bagging","SGD","GaussianNB"] + (["XGBoost"] if HAS_XGB else []))
    use_smote = st.checkbox("Use SMOTE (class imbalance)", value=True)
    target_col = st.text_input("Target column name", value="Churn")

pipeline = None
trained = False
df_upload = None

if load_existing:
    try:
        pipeline = joblib.load("churn_pipeline.joblib")
        st.success("Loaded existing pipeline: churn_pipeline.joblib")
    except Exception as e:
        st.error(f"Could not load pipeline: {e}")

uploaded = st.file_uploader("📤 Upload CSV dataset (e.g., Telco Customer Churn)", type=["csv"])

if uploaded is not None:
    try:
        df_upload = read_csv(uploaded)
        st.write("Data shape:", df_upload.shape)
        st.dataframe(df_upload.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if df_upload is not None:
    with st.expander("🔧 Clean & Preview", expanded=True):
        df_clean = clean_dataframe(df_upload.copy())
        st.write("After cleaning:", df_clean.shape)
        st.dataframe(df_clean.head())

    if pipeline is None and target_col not in df_clean.columns:
        st.warning(f"Target column '{target_col}' not found. You can still load an existing pipeline from the sidebar, or rename your target column to '{target_col}'.")

    if (pipeline is None) and (target_col in df_clean.columns):
        if st.button("🚀 Train pipeline now"):
            with st.spinner("Training..."):
                pipeline, metrics, X_like = train_pipeline(df_clean, target_col, model_name, use_smote)
                trained = True
                st.success(f"Trained {model_name}. Accuracy: {metrics['accuracy']:.4f}")
                st.text(metrics["report"])
                # Save pipeline to file for reuse
                try:
                    joblib.dump(pipeline, "churn_pipeline.joblib")
                    st.info("Saved pipeline to 'churn_pipeline.joblib'.")
                except Exception as e:
                    st.warning(f"Could not save pipeline: {e}")
    else:
        # If pipeline is already loaded
        X_like = df_clean.drop(columns=[target_col]) if (target_col in df_clean.columns) else df_clean

    if pipeline is not None or trained:
        st.markdown("---")
        st.subheader("🔎 Predict")
        form_df = make_prediction_form(pd.concat([X_like.head(200)], axis=0), target_col)
        if st.button("Predict churn"):
            try:
                pred = pipeline.predict(form_df)[0]
                proba = None
                if hasattr(pipeline, "predict_proba"):
                    proba = float(pipeline.predict_proba(form_df)[0,1])
                label = "🚨 CHURN" if int(pred)==1 else "✅ NOT CHURN"
                st.write("**Prediction:**", label)
                if proba is not None:
                    st.write(f"**Churn Probability:** {proba*100:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Download button for the saved pipeline if exists
import os
if os.path.exists("churn_pipeline.joblib"):
    with open("churn_pipeline.joblib", "rb") as f:
        st.download_button("⬇️ Download trained pipeline", data=f, file_name="churn_pipeline.joblib")
