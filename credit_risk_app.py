import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# Set page config
st.set_page_config(page_title="Credit Risk Prediction System", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Assuming the data is in the same directory as the app
    df = pd.read_csv('credit_risk_dataset.csv')
    
    # Handle missing values
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    
    # Remove outliers (e.g., age > 100)
    df = df[df['person_age'] <= 100]
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Define features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                      'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler, le, numerical_cols, categorical_cols

# Function to train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('credit_risk_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return model, report, cm, X_test, y_test

# Main app
def main():
    st.title("AI-Driven Credit Risk Prediction System")
    st.markdown("Developed for Grant Thornton Bharat - Summer Trainee Project")
    
    # Load and preprocess data
    X, y, scaler, le, numerical_cols, categorical_cols = load_and_preprocess_data()
    
    # Train model
    model, report, cm, X_test, y_test = train_model(X, y)
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Model Performance", "Predict Credit Risk"])
    
    if page == "Home":
        st.header("Welcome to the Credit Risk Prediction System")
        st.write("""
        This application uses AI to predict credit risk based on customer data. 
        The system is trained on a dataset containing customer information such as age, income, loan amount, and credit history.
        
        ### Features:
        - Predicts likelihood of loan default
        - Provides detailed model performance metrics
        - User-friendly interface for inputting customer data
        - Built with Random Forest Classifier for robust predictions
        
        Navigate using the sidebar to view model performance or make predictions.
        """)
        
        # Display dataset sample
        st.subheader("Sample Data")
        st.dataframe(pd.read_csv('credit_risk_dataset.csv').head())
    
    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        
        # Display classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        st.pyplot(fig)
    
    elif page == "Predict Credit Risk":
        st.header("Predict Credit Risk")
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("Enter Customer Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
                person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
                person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=5.0)
                loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
                
            with col2:
                person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
                loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 
                                                         'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
                loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F'])
                cb_person_default_on_file = st.selectbox("Default on File", ['Y', 'N'])
            
            loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0)
            loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2)
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Prepare input data
                input_data = pd.DataFrame({
                    'person_age': [person_age],
                    'person_income': [person_income],
                    'person_home_ownership': [person_home_ownership],
                    'person_emp_length': [person_emp_length],
                    'loan_intent': [loan_intent],
                    'loan_grade': [loan_grade],
                    'loan_amnt': [loan_amnt],
                    'loan_int_rate': [loan_int_rate],
                    'loan_percent_income': [loan_percent_income],
                    'cb_person_default_on_file': [cb_person_default_on_file],
                    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
                })
                
                # Encode categorical variables
                for col in categorical_cols:
                    input_data[col] = le.fit_transform(input_data[col])
                
                # Scale numerical features
                input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
                
                # Make prediction
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
                
                # Display results
                st.subheader("Prediction Results")
                if prediction[0] == 1:
                    st.error(f"High Risk: Customer is likely to default (Probability: {probability:.2%})")
                else:
                    st.success(f"Low Risk: Customer is unlikely to default (Probability: {probability:.2%})")
                
                # Display input summary
                st.subheader("Input Summary")
                st.write(input_data)

if __name__ == '__main__':
    main()