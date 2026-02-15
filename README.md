# E-commerce Customer Churn Prediction

## Problem Statement
Customer churn is a critical challenge for e-commerce businesses, as retaining existing customers is more cost-effective than acquiring new ones. This project aims to predict whether a customer will churn (leave the platform) based on their behavioral and demographic features. By identifying at-risk customers early, businesses can implement targeted retention strategies to reduce churn rates and improve customer lifetime value.

## Dataset Description
**Dataset Name:** E-commerce Customer Churn Dataset  
**Source:** https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data

**Dataset Characteristics:**
- **Total Instances:** 5630 (3774 non null entries were considered for training)
- **Total Features:** 18
- **Target Variable:** Churn (Binary: 0 = No Churn, 1 = Churn)
- **Feature Types:** Mix of numerical and categorical features

**Key Features:**
1. **Tenure** - Duration of customer relationship (months)
2. **WarehouseToHome** - Distance from warehouse to customer's home
3. **HourSpendOnApp** - Average hours spent on the mobile app
4. **NumberOfDeviceRegistered** - Number of devices registered by customer
5. **PreferedOrderCat** - Preferred order category (Fashion, Grocery, etc.)
6. **SatisfactionScore** - Customer satisfaction rating (1-5)
7. **Complain** - Number of complaints registered
8. **OrderAmountHikeFromlastYear** - Percentage increase in order amount
9. **CouponUsed** - Number of coupons used
10. **OrderCount** - Total number of orders placed
11. **DaySinceLastOrder** - Days since the last order
12. **CashbackAmount** - Total cashback amount received
13. **Gender** - Customer gender
14. **MaritalStatus** - Marital status of customer
15. **PreferredLoginDevice** - Preferred device for login
16. **PreferredPaymentMode** - Preferred payment method
17. **CityTier** - City tier classification (1, 2, or 3)
18. **NumberOfAddress** - Number of addresses registered

## Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 81.06% | 0.886 | 0.413 | 0.804 | 0.546 | 0.480 |
| Decision Tree | 85.56% | 0.910 | 0.493 | 0.701 | 0.579 | 0.506 |
| kNN | 79.47% | 0.944 | 0.398 | 0.879 | 0.548 | 0.496 |
| Naive Bayes | 24.11% | 0.805 | 0.155 | 0.981 | 0.268 | 0.114 |
| Random Forest (Ensemble) | 95.23% | 0.983 | 0.859 | 0.794 | 0.825 | 0.798 |
| XGBoost (Ensemble) | 93.38% | 0.954 | 0.806 | 0.701 | 0.750 | 0.714 |



## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Moderate accuracy (~81%) with good recall (~80%) but low precision (~41%), meaning it detects churn reasonably well but produces many false positives. AUC (~0.89) indicates decent class separation. |
| **Decision Tree** | Slightly higher accuracy (~85.6%) than logistic regression but lower recall (~70%). Precision (~49%) improved slightly. AUC (~0.91) suggests good classification capability but not the top performer. |
| **kNN** | Lower accuracy (~79%) but very high recall (~88%), indicating strong churn detection with many false positives due to lower precision (~40%). AUC (~0.94) is strong. |
| **Naive Bayes** | Very low accuracy (~24%) despite extremely high recall (~98%). Precision (~15%) is poor, meaning excessive churn prediction. MCC (~0.11) confirms weak predictive reliability. |
| **Random Forest (Ensemble)** | Best overall performer with highest accuracy (~95%), AUC (~0.98), strong precision (~86%), balanced recall (~79%), highest F1 (~0.83) and MCC (~0.80). Most reliable model overall. |
| **XGBoost (Ensemble)** | Second-best model with high accuracy (~93%) and strong AUC (~0.95). Good precision (~81%) but slightly lower recall (~70%) compared to Random Forest. Strong overall performance. |

### Summary
- **Best Model:** Random Forest  
- **Runner-up:** XGBoost  
- **Weakest Model:** Naive Bayes  
- Other models show trade-offs between precision and recall.


## Streamlit Application Features

The deployed web application provides an interactive interface for customer churn prediction with the following features:

1. **Single Customer Prediction**
   - Input customer details through sidebar controls
   - Select from 6 trained ML models
   - Get instant churn risk prediction with probability

2. **Batch Prediction**
   - Upload CSV file with multiple customer records
   - Download sample test data template
   - View prediction results in a scrollable table
   - Generate comprehensive performance report

3. **Model Performance Metrics**
   - Accuracy, AUC, Precision, Recall, F1 Score, MCC
   - Visual confusion matrix
   - Expandable performance report section

4. **User-Friendly Interface**
   - Clean and intuitive design
   - Responsive layout
   - Real-time predictions
   - Visual feedback for predictions

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/debashisgho/bits-ml-classification-model.git
cd bits-ml-classification-model
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
bits-ml-classification-model/
│
├── app.py                          # Main Streamlit application
├── model_training.ipynb            # Jupyter notebook for model training
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── raw/
│   │   ├── raw_ecom_customer_churn.csv
│   │   └── Full_test_raw.csv
│   └── processed/
│       ├── train_data.pkl
│       └── test_data.pkl
│
└── model/                   # Trained model artifacts
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── preprocessor.pkl
    └── features.pkl
```

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning models and preprocessing
- **XGBoost** - Gradient boosting implementation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization

## Model Training Process

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling and normalization
   - Train-test split

2. **Model Training**
   - Training 6 different classification models
   - Hyperparameter tuning (if applicable)
   - Cross-validation for robust evaluation

3. **Model Evaluation**
   - Calculating 6 evaluation metrics for each model
   - Generating confusion matrices
   - Comparing model performances

4. **Model Persistence**
   - Saving trained models using pickle
   - Saving preprocessor for consistent transformations

## Deployment

The application is deployed on **Streamlit Community Cloud** and can be accessed at:

**Live App URL:** https://debashisgho-ml-classification-model.streamlit.app/

## Usage Instructions

### For Single Prediction:
1. Adjust customer profile parameters in the sidebar
2. Select a model from the dropdown
3. Click "Predict Churn" button
4. View the prediction result with probability

### For Batch Prediction:
1. Download the sample CSV template (optional)
2. Prepare your CSV file with customer data
3. Upload the CSV file
4. Select a model
5. Click "Run Batch Prediction"
6. View results and performance metrics

## Future Enhancements

- Add feature importance visualization
- Implement SHAP values for model explainability
- Add model comparison dashboard
- Include ROC curve and Precision-Recall curve
- Add data quality checks and validation
- Implement real-time model retraining capability

## Author

**Debashis Ghosh**  
Student ID: 2025AA05806  
BITS Pilani - M.Tech (AI/ML)  
Machine Learning - Assignment 2

## License

This project is created for academic purposes as part of BITS Pilani M.Tech coursework.

## Acknowledgments

- BITS Pilani Work Integrated Learning Programmes Division
- Dataset source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data
- Streamlit Community for deployment platform
