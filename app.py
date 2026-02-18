import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model and Artifacts
try:
    artifacts = joblib.load('loan_model.pkl')
    # Check if 'models' key exists (new format) or fall back to 'model' (old format)
    if 'models' in artifacts:
        models = artifacts['models']
        model_name = "Multiple Models (NB, LR, KNN)"
    else:
        models = {'Best Model': artifacts['model']}
        model_name = artifacts.get('model_name', 'Best Model')
        
    num_imputer = artifacts['num_imputer']
    cat_imputer = artifacts['cat_imputer']
    le_education = artifacts['le_education']
    ohe = artifacts['ohe']
    scaler = artifacts['scaler']
    numerical_cols = artifacts['numerical_cols'].tolist()
    # feature_cat_cols = artifacts['feature_cat_cols'] 
    oh_cols = artifacts['oh_cols']
    X_columns = artifacts['X_columns']
    target_classes = artifacts.get('target_classes', ['Rejected', 'Approved'])
except FileNotFoundError:
    st.error("Model file 'loan_model.pkl' not found. Please run 'train_model.py' first.")
    st.stop()

st.title(f"Loan Approval Prediction App")
st.caption(f"Powered by {model_name}")

# Initialize Session State for inputs
if 'applicant_income' not in st.session_state:
    st.session_state['applicant_income'] = 0.0
if 'coapplicant_income' not in st.session_state:
    st.session_state['coapplicant_income'] = 0.0
if 'loan_amount' not in st.session_state:
    st.session_state['loan_amount'] = 0.0
if 'loan_term' not in st.session_state:
    st.session_state['loan_term'] = 12.0
if 'credit_score' not in st.session_state:
    st.session_state['credit_score'] = 300.0
if 'age' not in st.session_state:
    st.session_state['age'] = 18
if 'dependents' not in st.session_state:
    st.session_state['dependents'] = 0
if 'existing_loans' not in st.session_state:
    st.session_state['existing_loans'] = 0
if 'savings' not in st.session_state:
    st.session_state['savings'] = 0.0
if 'collateral_value' not in st.session_state:
    st.session_state['collateral_value'] = 0.0
if 'dti_ratio' not in st.session_state:
    st.session_state['dti_ratio'] = 0.0
if 'gender' not in st.session_state:
    st.session_state['gender'] = "Male"
if 'marital_status' not in st.session_state:
    st.session_state['marital_status'] = "Single"
if 'education_level' not in st.session_state:
    st.session_state['education_level'] = "Not Graduate"
if 'employment_status' not in st.session_state:
    st.session_state['employment_status'] = "Unemployed"
if 'employer_category' not in st.session_state:
    st.session_state['employer_category'] = "Unemployed"
if 'property_area' not in st.session_state:
    st.session_state['property_area'] = "Rural"
if 'loan_purpose' not in st.session_state:
    st.session_state['loan_purpose'] = "Personal"


# Sample Data Buttons
col_sample1, col_sample2 = st.columns(2)

with col_sample1:
    if st.button("Fill Approved Sample"):
        st.session_state['applicant_income'] = 50000.0
        st.session_state['coapplicant_income'] = 20000.0
        st.session_state['loan_amount'] = 100000.0
        st.session_state['loan_term'] = 360.0
        st.session_state['credit_score'] = 750.0
        st.session_state['age'] = 35
        st.session_state['dependents'] = 0
        st.session_state['existing_loans'] = 0
        st.session_state['savings'] = 5000.0
        st.session_state['collateral_value'] = 20000.0
        st.session_state['dti_ratio'] = 0.2
        st.session_state['gender'] = "Male"
        st.session_state['marital_status'] = "Married"
        st.session_state['education_level'] = "Graduate"
        st.session_state['employment_status'] = "Salaried"
        st.session_state['employer_category'] = "MNC"
        st.session_state['property_area'] = "Urban"
        st.session_state['loan_purpose'] = "Home"
        st.rerun()

with col_sample2:
    if st.button("Fill Rejected Sample"):
        st.session_state['applicant_income'] = 2000.0
        st.session_state['coapplicant_income'] = 0.0
        st.session_state['loan_amount'] = 500000.0
        st.session_state['loan_term'] = 120.0
        st.session_state['credit_score'] = 400.0
        st.session_state['age'] = 22
        st.session_state['dependents'] = 5
        st.session_state['existing_loans'] = 2
        st.session_state['savings'] = 0.0
        st.session_state['collateral_value'] = 0.0
        st.session_state['dti_ratio'] = 0.8
        st.session_state['gender'] = "Male"
        st.session_state['marital_status'] = "Single"
        st.session_state['education_level'] = "Not Graduate"
        st.session_state['employment_status'] = "Unemployed"
        st.session_state['employer_category'] = "Unemployed"
        st.session_state['property_area'] = "Rural"
        st.session_state['loan_purpose'] = "Personal"
        st.rerun()


st.write("Enter the details below to check loan eligibility.")

# Input Form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Define lists for selectboxes to find index
        gender_opts = ["Male", "Female"]
        marital_opts = ["Married", "Single"]
        education_opts = ["Graduate", "Not Graduate"]
        employment_opts = ["Salaried", "Self-employed", "Contract", "Unemployed"]
        employer_opts = ["Private", "Government", "MNC", "Business", "Unemployed"]
        property_opts = ["Urban", "Semiurban", "Rural"]
        purpose_opts = ["Home", "Car", "Personal", "Business", "Education"]

        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=st.session_state['applicant_income'])
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=st.session_state['coapplicant_income'])
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=st.session_state['loan_amount'])
        loan_term = st.number_input("Loan Term (Months)", min_value=12.0, value=st.session_state['loan_term'])
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=st.session_state['credit_score'])
        age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state['age'])
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=st.session_state['dependents'])

    with col2:
        existing_loans = st.number_input("Existing Loans", min_value=0, value=st.session_state['existing_loans'])
        savings = st.number_input("Savings Account Balance", min_value=0.0, value=st.session_state['savings'])
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=st.session_state['collateral_value']) 
        dti_ratio = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=1.0, value=st.session_state['dti_ratio'])
        
    st.subheader("Personal Details")
    col3, col4 = st.columns(2)
    with col3:
        try:
             gender_idx = gender_opts.index(st.session_state['gender'])
        except: gender_idx = 0
        gender = st.selectbox("Gender", gender_opts, index=gender_idx)
        
        try:
             marital_idx = marital_opts.index(st.session_state['marital_status'])
        except: marital_idx = 0
        marital_status = st.selectbox("Marital Status", marital_opts, index=marital_idx)
        
        try:
             edu_idx = education_opts.index(st.session_state['education_level'])
        except: edu_idx = 0
        education_level = st.selectbox("Education Level", education_opts, index=edu_idx)
        
    with col4:
        try:
             emp_idx = employment_opts.index(st.session_state['employment_status'])
        except: emp_idx = 0
        employment_status = st.selectbox("Employment Status", employment_opts, index=emp_idx)
        
        try:
             employer_idx = employer_opts.index(st.session_state['employer_category'])
        except: employer_idx = 0
        employer_category = st.selectbox("Employer Category", employer_opts, index=employer_idx)
        
        try:
             prop_idx = property_opts.index(st.session_state['property_area'])
        except: prop_idx = 0
        property_area = st.selectbox("Property Area", property_opts, index=prop_idx)
        
        try:
             loan_idx = purpose_opts.index(st.session_state['loan_purpose'])
        except: loan_idx = 0
        loan_purpose = st.selectbox("Loan Purpose", 
                                    purpose_opts, index=loan_idx)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame from input
    input_data = pd.DataFrame({
        'Applicant_Income': [applicant_income],
        'Coapplicant_Income': [coapplicant_income],
        'Loan_Amount': [loan_amount],
        'Loan_Term': [loan_term],
        'Credit_Score': [credit_score],
        'Age': [age],
        'Dependents': [dependents],
        'Existing_Loans': [existing_loans],
        'Savings': [savings],
        'Collateral_Value': [collateral_value],
        'DTI_Ratio': [dti_ratio],
        'Gender': [gender],
        'Marital_Status': [marital_status],
        'Education_Level': [education_level],
        'Employment_Status': [employment_status],
        'Employer_Category': [employer_category],
        'Property_Area': [property_area],
        'Loan_Purpose': [loan_purpose],
        # 'Applicant_ID': [0] 
    })

    # --- Preprocessing ---
    
    # 1. Imputation (Skipped)
    
    # 2. Label Encoding
    try:
        input_data["Education_Level"] = le_education.transform(input_data["Education_Level"])
    except ValueError as e:
         st.error(f"Encoding Error: {e}")
         st.stop()
         
    # 3. One-Hot Encoding
    try:
        encoded_features = ohe.transform(input_data[oh_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(oh_cols))
        encoded_df.index = input_data.index 
        
        input_data_processed = pd.concat([input_data.drop(columns=oh_cols), encoded_df], axis=1)
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        st.stop()

    # 4. Scaling
    try:
        model_input = input_data_processed[X_columns]
        model_input_scaled = scaler.transform(model_input)
    except KeyError as e:
        st.error(f"Feature Mismatch: {e}")
        st.write("Current columns:", input_data_processed.columns.tolist())
        st.write("Expected columns:", X_columns.tolist())
        st.stop()
    
    # 5. Prediction (All Models)
    st.divider()
    st.subheader("Results")
    
    # Create columns for each model result
    cols = st.columns(len(models))
    
    for i, (name, model) in enumerate(models.items()):
        
        prediction = model.predict(model_input_scaled)
        
        # Decode prediction
        result_index = prediction[0]
        st.write(f"Debug: Model {name} predicted index {result_index}")
        if len(target_classes) > 1:
            result_label = target_classes[result_index]
            if str(result_label).lower() in ['yes', 'y', '1', 'approved']:
                 result = "Approved"
                 status = "success"
            elif str(result_label).lower() in ['no', 'n', '0', 'rejected']:
                 result = "Rejected"
                 status = "error"
            else:
                 result = str(result_label)
                 status = "info"
        else:
            # Fallback
            result = "Approved" if result_index == 1 else "Rejected"
            status = "success" if result == "Approved" else "error"
        
        with cols[i]:
            st.markdown(f"**{name}**")
            if status == "success":
                st.success(f"{result}")
            else:
                st.error(f"{result}")

