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
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "./output-model"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

header_col1, header_col2, header_col3 = st.columns([1, 6, 1])

with header_col2:
    st.markdown("""
#### 📉 Ecommerce Customer Churn Prediction
<small>Predict churn risk using ML models trained on customer behavior.</small>  
<small><i>BITS Pilani – ML Assignment 2 | Debashis Ghosh (2025AA05806)</i></small>
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
    st.sidebar.markdown("## Customer Profile")

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
st.subheader("Prediction Interface")

model_options = [
    f.replace(".pkl", "").replace("_", " ").title()
    for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and f not in ["preprocessor.pkl", "features.pkl"]
]

selected_model = st.selectbox("Select Model", model_options).lower()

if st.button("Predict Churn"):
    model = load_model(selected_model)
    input_scaled = preprocessor.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba") else 0
    )

    if prediction == 1:
        st.error(f"High Churn Risk (Probability: {probability:.2%})")
    else:
        st.success(f"Low Churn Risk (Probability: {probability:.2%})")


# -------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------
st.divider()
st.subheader("Batch Prediction")

# Initialize variables to avoid NameError
y_true = None
y_pred = None


uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    if st.button("Run Batch Prediction"):
        model = load_model(selected_model)

        y_true = data.pop("Churn") if "Churn" in data.columns else None
        customer_ids = data.pop("CustomerID") if "CustomerID" in data.columns else None


        X_scaled = preprocessor.transform(data)
        y_pred = model.predict(X_scaled)

        st.success("Predictions generated")

        result = pd.DataFrame({
            "Customer ID": customer_ids,
            "Actual": y_true,
            "Predicted": y_pred
        })

        st.dataframe(result)

        # Metrics

if y_true is not None:
    with st.expander("Model Performance Report", expanded=False):

        # --- Metrics ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        c2.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
        c3.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
        c4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2%}")

        st.divider()

        # --- Confusion Matrix + Report ---
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4.5, 3.5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 9, "weight": "bold"},
                linewidths=0.5,
                linecolor="gray",
                ax=ax
            )

            ax.set_title("Confusion Matrix", fontsize=11, weight="bold", pad=10)
            ax.set_xlabel("Predicted Label", fontsize=9)
            ax.set_ylabel("Actual Label", fontsize=9)

            ax.set_xticklabels(["No Churn", "Churn"], fontsize=8)
            ax.set_yticklabels(["No Churn", "Churn"], fontsize=8, rotation=0)

            plt.tight_layout()
            st.pyplot(fig)


