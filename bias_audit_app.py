import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    true_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference
)
import shap
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Bias Audit Dashboard", layout="wide")
st.title("🧪 Fairness & Bias Audit Dashboard")

# --- UPLOAD DATA ---
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select target variable
    target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    
    # Optional: Drop ID column or similar
    id_col = st.sidebar.selectbox("🆔 Drop Column (Optional)", ['None'] + list(df.columns))
    if id_col != 'None':
        df = df.drop(columns=[id_col])

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # --- SELECT PROTECTED ATTRIBUTE ---
    protected_options = [col for col in df.columns if col != target_col]
    protected_attr = st.sidebar.selectbox("🛡️ Select Protected Attribute", protected_options)

    # Check encoded column name for protected attribute
    protected_attr_encoded = [col for col in df_encoded.columns if protected_attr in col]
    if not protected_attr_encoded:
        st.warning("⚠️ No encoded column found for the selected protected attribute.")
        st.stop()
    protected_attr_encoded_name = protected_attr_encoded[0]

    # --- TRAIN/TEST SPLIT ---
    X = df_encoded.drop(columns=[target_col, protected_attr_encoded_name])
    y = df_encoded[target_col]
    A = df_encoded[protected_attr_encoded_name]

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.3, random_state=42)

    # --- TRAIN MODEL ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("✅ Model Performance")
    st.metric(label="Accuracy", value=f"{accuracy:.4f}")

    # --- FAIRNESS METRICS ---
    st.subheader("📊 Fairness Metrics")
    metrics_dict = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "TPR": true_positive_rate,
        "TNR": true_negative_rate
    }

    metric_frame = MetricFrame(
        metrics=metrics_dict,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    )

    st.write("Group-wise Metrics:")
    st.dataframe(metric_frame.by_group)

    st.write("📉 Group Disparity Differences:")
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)

    st.metric("Demographic Parity Difference", f"{dp_diff:.4f}")
    st.metric("Equalized Odds Difference", f"{eo_diff:.4f}")

    # --- PLOT METRICS ---
    st.subheader("📈 Fairness Metric Plots")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    metric_frame.by_group["accuracy"].plot.bar(ax=axes[0, 0], title="Accuracy")
    metric_frame.by_group["selection_rate"].plot.bar(ax=axes[0, 1], title="Selection Rate")
    metric_frame.by_group["TPR"].plot.bar(ax=axes[1, 0], title="True Positive Rate")
    metric_frame.by_group["TNR"].plot.bar(ax=axes[1, 1], title="True Negative Rate")
    st.pyplot(fig)

    # --- SHAP EXPLANATION ---
    st.subheader("🧠 SHAP Feature Importance")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Display SHAP summary plot
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values[1], X_test)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP Analysis Failed: {e}")
else:
    st.info("👈 Please upload a CSV file to begin.")

