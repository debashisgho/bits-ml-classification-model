# app.py
import streamlit as st
import pandas as pd
import pickle
import os
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

# --- CUSTOM CSS ---
# st.markdown(unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)


# st.title("Ecommerce Customer Churn Prediction")
# st.title("CHANGE HEADER")
# st.markdown("**BITS Pilani - ML Assignment 2** | Model: SMOTE Enhanced")
# st.markdown("**Change markdown")
st.markdown("""
<div class="card">
    <h1>📉 Ecommerce Customer Churn Prediction</h1>
    <p style="font-size:1.05rem;color:#475569;">
        Predict churn risk using ML models trained on customer behavior, transactions,
        and engagement patterns.
    </p>
    <p><b>BITS Pilani – ML Assignment 2 | BITS ID # 2025AA05806 | Debashis Ghosh</b></p>
</div>
""", unsafe_allow_html=True)
# --- 1. LOAD ARTIFACTS ---

MODEL_PATH = "./output-model"

if not os.path.exists(MODEL_PATH):
    st.error(f"Folder '{MODEL_PATH}' not found. Please run the notebook first.")
    st.stop()


# @st.cache_resource
def load_resources():
    try:
        preprocessor = pickle.load(open(f"{MODEL_PATH}/preprocessor.pkl", "rb"))
        feature_names = pickle.load(open(f"{MODEL_PATH}/features.pkl", "rb"))
        
        ##############debug

        # encoder = preprocessor.named_transformers_["cat"]["onehot"]
        # cat_cols = preprocessor.transformers_[1][2]
        # idx = cat_cols.index("CityTier")

        # st.write("CityTier categories learned:", encoder.categories_[idx])
        # st.write("Type:", [type(x) for x in encoder.categories_[idx]])
        # # st.write("Streamlit CityTier value:", input_df["CityTier"].iloc[0])
        # # st.write("Type:", type(input_df["CityTier"].iloc[0]))


        ########################
        return preprocessor, feature_names
    except FileNotFoundError:
        return None, None


preprocessor, feature_names = load_resources()


if not preprocessor:
    st.error(
        "Artifacts missing. Run your training notebook to generate 'scaler.pkl' and 'features.pkl'."
    )
    st.stop()

# --- 2. SIDEBAR INPUTS ---
st.sidebar.markdown("## 🧾 Customer Profile")
st.sidebar.markdown("Fill customer details to assess churn risk")
st.sidebar.divider()
# st.sidebar.header("Customer Parameters")


numeric_cols = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "NumberOfAddress", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "SatisfactionScore", "Complain", "DaySinceLastOrder", "CashbackAmount"
]


def user_input_features():

    st.sidebar.markdown("### 👤 Customer Basics")

    gender = st.sidebar.selectbox(
        "Gender",
        [
            "Female" , "Male"
        ]
    )

        # --- Marital Status (Dropdown) ---
    maritalStatus = st.sidebar.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single"],
        index=2                # default = "Single"
    )

    cityTier = st.sidebar.slider("City Tier", 1,3,1 )

    st.sidebar.markdown("### 📱 Engagement & Usage")

    preferredLoginDevice = st.sidebar.selectbox(
        "Logon Device", ['Mobile Phone', 'Computer']
    )

    registeredDevices = st.sidebar.slider("Registered Devices", 1, 10, 1)

    hoursOnApp = st.sidebar.slider("App Time (Hours)", 0, 10, 3)

    st.sidebar.markdown("### 🛒 Orders & Payments")

    preferredPaymentMethod = st.sidebar.selectbox(
        "Payment Method",
        [
            "Cash on Delivery",
            "Credit Card",
            "Debit Card",
            "E wallet",
            "UPI"
        ]
    )

    
    preferredOrderCategory = st.sidebar.selectbox(
        "Preferred Order Category",
        [
            "Fashion",
            "Grocery",
            "Laptop & Accessory",
            "Mobile Phone",
            "Others"
        ]
    )


    # --- Coupons Used (0 to 30) ---
    couponsUsed = st.sidebar.slider(
        "Coupons Used",
        min_value=0,
        max_value=30,
        value=0,
        step=1
    )

    orderCount = st.sidebar.slider(
        "Order Count",
        min_value=1,
        max_value=30,
        value=2,
        step=1
    )

    daySinceLastOrder = st.sidebar.slider(
        "Last Order (Days)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    cashbackAmount = st.sidebar.slider(
        "Cashback (Rs)",
        min_value=0,
        max_value=400,
        value=100,
        step=1
    )

    
    # --- Order Amount Hike From Last Year (0 to 100000) ---
    orderAmountHikeFromLastYear = st.sidebar.slider(
        "Order Amount Hike From Last Year",
        min_value=0,
        max_value=100000,
        value=0,
        step=1000              # adjust granularity if needed
    )

    # Helper function to match training columns
    tenure = st.sidebar.slider("Tenure (Months)", 0, 70, 2)
   

    warehouseHomeDistance = st.sidebar.slider("Warehouse distance (Kms)", 1, 150, 20)


    satisfactionScore = st.sidebar.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3,               # default
        step=1
    )



    # --- Number of Addresses (1 to 30) ---
    numberOfAddress = st.sidebar.slider(
        "Number of Addresses",
        min_value=1,
        max_value=30,
        value=1,
        step=1
    )

    # --- Complaint Count (0 to 10) ---
    complaintCount = st.sidebar.slider(
        "Complaint Count",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )



 


    data = {
        "Tenure": tenure,
        "WarehouseToHome": warehouseHomeDistance,
        "HourSpendOnApp": hoursOnApp,
        "NumberOfDeviceRegistered": registeredDevices,
        "NumberOfAddress": numberOfAddress,
        "OrderAmountHikeFromlastYear": orderAmountHikeFromLastYear,
        "CouponUsed": couponsUsed,
        "OrderCount": orderCount,
        "SatisfactionScore": satisfactionScore,
        "Complain": complaintCount,
        "DaySinceLastOrder": daySinceLastOrder,
        "CashbackAmount": cashbackAmount,
        "Gender": gender,
        "PreferredLoginDevice": preferredLoginDevice,
        "CityTier": cityTier,
        "PreferredPaymentMode": preferredPaymentMethod,
        "PreferedOrderCat": preferredOrderCategory,
        "MaritalStatus": maritalStatus

    }

    input_df = pd.DataFrame([data])


    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")


    return pd.DataFrame(data, index=[0])



input_df = user_input_features()
# st.subheader("DEBUG — Raw Input")
# st.write(input_df)
# st.write(input_df.dtypes)

# --- 3. PREDICTION ---
st.subheader("Prediction Interface")


model_list = [
    f.replace(".pkl", "").replace("_", " ").title()
    for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and f not in ["preprocessor.pkl", "features.pkl"]
]

selected_model = st.selectbox("Select Model", model_list)

if st.button("Predict Churn"):
    # Preprocess
    # input_encoded = pd.get_dummies(input_df)
    # input_encoded = input_encoded.reindex(columns= numeric_cols, fill_value=0)
    # input_scaled = scaler.transform(input_encoded)
    input_scaled = preprocessor.transform(input_df)

    # st.write(input_scaled)
    ################debug

    # Xt = preprocessor.transform(input_df)

    # if hasattr(Xt, "toarray"):
    #     Xt = Xt.toarray()
    # st.subheader("DEBUG — Transformed Output")
    # st.write("Shape:", Xt.shape)
    # st.write("Row sum:", Xt.sum())
    # st.write("Non-zero count:", (Xt != 0).sum())

    # feature_names_now = preprocessor.get_feature_names_out()

    # active_features = [
    #     feature_names_now[i]
    #     for i in range(len(feature_names_now))
    #     if Xt[0, i] != 0
    # ]

    # st.subheader("DEBUG — Active Features")
    # st.write(active_features)
    ################debug###########


    # ###########################################debug
    # # print("Model expects:", getattr(model, "n_features_in_", None))  # should print 33
    # Xt = preprocessor.transform(input_df)
    # if hasattr(Xt, "toarray"):
    #     Xt = Xt.toarray()
    # print("Preprocessor currently outputs:", Xt.shape[1])            # prints 30 per your list

    # try:
    #     feat_names_now = preprocessor.get_feature_names_out()
    #     print("Feature names now (len={}):".format(len(feat_names_now)))
    #     for f in feat_names_now: print(f)
    # except Exception as e:
    #     print("Could not get feature names:", e)

    # ################################################



    # Load Model
    selected_model_full_name = selected_model.replace(" ","_") +".pkl"
    model = pickle.load(open(f"{MODEL_PATH}/{selected_model_full_name}", "rb"))

    ## debug######333

    # st.subheader("DEBUG — Model Expectation")
    # st.write("Model expects:",
    #     getattr(model, "n_features_in_", None))
    ###################

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else 0
    )


    # Display
    if pred == 1:
        st.error(f"⚠️ High Churn Risk (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Safe - Less Churn Risk (Probability: {prob:.2%})")

# --- 4. BATCH UPLOAD (With Requirements C & D) ---
st.divider()
st.subheader("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", test_data.head(10))

    if st.button("Run Batch Prediction"):
        test_data_copy = test_data.copy()
        # Basic Cleaning
        if "CustomerID" in test_data.columns:
            test_data = test_data.drop("CustomerID", axis=1)


        # Prepare X and y
        if "Churn" in test_data.columns:
            # Map Yes/No to 1/0
            y_true = test_data["Churn"]
            X_test = test_data.drop("Churn", axis=1)
        else:
            y_true = None
            X_test = test_data

        # Transform
        X_test_scaled = preprocessor.transform(X_test)
        # X_test_encoded = pd.get_dummies(X_test)
        # X_test_encoded = X_test_encoded.reindex(columns=feature_names, fill_value=0)
        # X_test_scaled = scaler.transform(X_test_encoded)

        selected_model_full_name = selected_model.replace(" ","_") +".pkl"
        model = pickle.load(open(f"{MODEL_PATH}/{selected_model_full_name}", "rb"))

        y_pred = model.predict(X_test_scaled)

#         # Predict
#         model = pickle.load(open(f"{MODEL_PATH}/{selected_model}", "rb"))
#         y_pred = model.predict(X_test_scaled)

        st.success("Predictions generated!")
        test_data["Prediction"] = y_pred
        test_data["CustomerID"] = test_data_copy["CustomerID"]


        st.dataframe(
        test_data[["CustomerID", "Churn", "Prediction"]]
        .rename(columns={
            "CustomerID": "Customer ID",
            "Churn": "Actual (Y_True)",
            "Prediction": "Predicted (Y_Pred)"
        })
)


        # --- [ADDED] EVALUATION METRICS (Requirements C & D) ---
        if y_true is not None:
            st.divider()
            st.subheader("📊 Model Performance Report")

            # 1. Metrics [Req C]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            m2.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
            m3.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
            m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2%}")

            # 2. Confusion Matrix & Classification Report [Req D]
            col_left, col_right = st.columns(2)

            with col_left:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col_right:
                st.write("**Classification Report**")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.info(
                "Note: Upload a CSV with a 'Churn' column to see evaluation metrics."
            )

