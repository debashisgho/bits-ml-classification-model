# app.py
import os
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "./model"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# Custom CSS for blue buttons and compact header
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #1E88E5;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1565C0;
        color: white;
    }
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
<div style='text-align: center; padding: 0.5rem 0 1rem 0;'>
    <h3 style='margin: 0;'>📉 Ecommerce Customer Churn Prediction</h3>
    <p style='color: #666; margin: 0.2rem 0 0 0; font-size: 0.9rem;'>Predict churn risk using ML models | <i>BITS Pilani – Debashis Ghosh (2025AA05806)</i></p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Folder '{MODEL_PATH}' not found.")
        st.stop()

    try:
        with open(f"{MODEL_PATH}/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open(f"{MODEL_PATH}/features.pkl", "rb") as f:
            feature_names = pickle.load(f)

        return preprocessor, feature_names

    except FileNotFoundError:
        st.error("Required artifacts missing.")
        st.stop()


preprocessor, feature_names = load_artifacts()


def load_model(model_name):
    model_file = model_name.replace(" ", "_") + ".pkl"
    with open(f"{MODEL_PATH}/{model_file}", "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
numeric_cols = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "NumberOfAddress",
    "OrderAmountHikeFromlastYear", "CouponUsed",
    "OrderCount", "SatisfactionScore", "Complain",
    "DaySinceLastOrder", "CashbackAmount"
]


def get_user_inputs():
    st.sidebar.markdown("### 👤 Customer Profile")
    st.sidebar.markdown("---")

    data = {
        "Gender": st.sidebar.selectbox("Gender", ["Female", "Male"]),
        "MaritalStatus": st.sidebar.selectbox(
            "Marital Status", ["Divorced", "Married", "Single"], index=2
        ),
        "CityTier": st.sidebar.slider("City Tier", 1, 3, 1),
        "PreferredLoginDevice": st.sidebar.selectbox(
            "Login Device", ["Mobile Phone", "Computer"]
        ),
        "PreferredPaymentMode": st.sidebar.selectbox(
            "Payment Method",
            ["Cash on Delivery", "Credit Card", "Debit Card", "E wallet", "UPI"]
        ),
        "PreferedOrderCat": st.sidebar.selectbox(
            "Order Category",
            ["Fashion", "Grocery", "Laptop & Accessory",
             "Mobile Phone", "Others"]
        ),
        "Tenure": st.sidebar.slider("Tenure", 0, 70, 2),
        "WarehouseToHome": st.sidebar.slider("Warehouse Distance", 1, 150, 20),
        "HourSpendOnApp": st.sidebar.slider("Hours on App", 0, 10, 3),
        "NumberOfDeviceRegistered": st.sidebar.slider("Devices", 1, 10, 1),
        "NumberOfAddress": st.sidebar.slider("Addresses", 1, 30, 1),
        "OrderAmountHikeFromlastYear":
            st.sidebar.slider("Order Amount Increase", 0, 100000, 0, 1000),
        "CouponUsed": st.sidebar.slider("Coupons Used", 0, 30, 0),
        "OrderCount": st.sidebar.slider("Order Count", 1, 30, 2),
        "SatisfactionScore": st.sidebar.slider("Satisfaction", 1, 5, 3),
        "Complain": st.sidebar.slider("Complaints", 0, 10, 0),
        "DaySinceLastOrder": st.sidebar.slider("Days Since Last Order", 0, 100, 5),
        "CashbackAmount": st.sidebar.slider("Cashback", 0, 400, 100),
    }

    df = pd.DataFrame([data])

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


input_df = get_user_inputs()

# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
st.markdown("### 🎯 Single Customer Prediction")
st.markdown("")

model_options = [
    f.replace(".pkl", "").replace("_", " ").title()
    for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and f not in ["preprocessor.pkl", "features.pkl"]
]

col1, col2 = st.columns([2, 1])
with col1:
    selected_model = st.selectbox(" Select Model", model_options).lower()
with col2:
    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("▶️ Predict Churn", use_container_width=True, type="primary")
 
if predict_btn:
    model = load_model(selected_model)
    input_scaled = preprocessor.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba") else 0
    )

    st.markdown("")
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk** — Probability: {probability:.2%}")
    else:
        st.success(f"✅ **Low Churn Risk** — Probability: {probability:.2%}")


# -------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Batch Prediction")
st.markdown("")

# Initialize variables to avoid NameError
y_true = None
y_pred = None

# Download sample data button
col_download, col_upload  = st.columns([1, 2])

with col_upload:
    uploaded_file = st.file_uploader("📁 Upload CSV file with customer data", type="csv")

with col_download:
    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
    sample_file_path = "./data/raw/Full_test_raw.csv"
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Sample",
                data=f,
                file_name="sample_test_data.csv",
                mime="text/csv",
                use_container_width=True
            )

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    with st.expander("📋 Preview Uploaded Data", expanded=False):
        st.dataframe(data.head(), use_container_width=True)

    if st.button("▶️ Run Batch Prediction", type="primary"):
        model = load_model(selected_model)

        y_true = data.pop("Churn") if "Churn" in data.columns else None
        customer_ids = data.pop("CustomerID") if "CustomerID" in data.columns else None


        X_scaled = preprocessor.transform(data)
        y_pred = model.predict(X_scaled)

        proba = model.predict_proba(X_scaled)
        pos_class_index = list(model.classes_).index(1)
        y_prob = proba[:, pos_class_index]

        # y_prob = (
        # model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        # )

        st.success("✅ Predictions generated successfully!")
        st.markdown("")

        result = pd.DataFrame({
            "Customer ID": customer_ids,
            "Actual": y_true,
            "Predicted": y_pred
        })

        st.markdown("**📈 Prediction Results**")
        st.dataframe(result, height=200, use_container_width=True)

        # Metrics

if y_true is not None:
    st.markdown("")
    with st.expander("📊 Model Performance Report", expanded=True):

        # --- Metrics ---
        st.markdown("**Performance Metrics**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        c2.metric("AUC", f"{roc_auc_score(y_true, y_prob):.2%}")
        c3.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
        c4.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
        c5.metric("F1 Score", f"{f1_score(y_true, y_pred):.2%}")
        c6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2%}")

        st.markdown("")
        st.markdown("---")

        # --- Confusion Matrix ---
        st.markdown("**📉 Confusion Matrix**")
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 12},
                linewidths=1,
                linecolor="white",
                ax=ax,
                square=True
            )

            ax.set_title("Confusion Matrix", fontsize=11, pad=12)
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("Actual Label", fontsize=10)

            ax.set_xticklabels(["No Churn", "Churn"], fontsize=9)
            ax.set_yticklabels(["No Churn", "Churn"], fontsize=9, rotation=0)

            plt.tight_layout()
            st.pyplot(fig)


