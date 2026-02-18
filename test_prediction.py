import pandas as pd
import numpy as np
import joblib

# Load Model and Artifacts
try:
    artifacts = joblib.load('loan_model.pkl')
    # Check if 'models' key exists (new format) or fall back to 'model' (old format)
    if 'models' in artifacts:
        models = artifacts['models']
        print(f"Loaded Models: {list(models.keys())}")
    else:
        models = {'Best Model': artifacts['model']}
        print(f"Loaded Single Model.")
        
    num_imputer = artifacts['num_imputer']
    cat_imputer = artifacts['cat_imputer']
    le_education = artifacts['le_education']
    ohe = artifacts['ohe']
    scaler = artifacts['scaler']
    numerical_cols = artifacts['numerical_cols'].tolist()
    oh_cols = artifacts['oh_cols']
    X_columns = artifacts['X_columns']
    target_classes = artifacts.get('target_classes', ['Rejected', 'Approved'])
    
except FileNotFoundError:
    print("Error: loan_model.pkl not found.")
    exit()

# Sample Input (Data that should be Approved/Rejected)
# Based on EDA, High Income + Good Credit + Low DTI -> Approved
input_data = pd.DataFrame({
    'Applicant_Income': [50000],
    'Coapplicant_Income': [20000],
    'Loan_Amount': [100000],
    'Loan_Term': [360],
    'Credit_Score': [750],
    'Age': [35],
    'Dependents': [0],
    'Existing_Loans': [0],
    'Savings': [5000],
    'Collateral_Value': [20000],
    'DTI_Ratio': [0.2],
    'Gender': ['Male'],
    'Marital_Status': ['Married'],
    'Education_Level': ['Graduate'],
    'Employment_Status': ['Salaried'],
    'Employer_Category': ['MNC'],
    'Property_Area': ['Urban'],
    'Loan_Purpose': ['Home']
})

print("\nInput Data:")
print(input_data)

# --- Preprocessing ---

# 1. Label Encoding
try:
    input_data["Education_Level"] = le_education.transform(input_data["Education_Level"])
except ValueError as e:
     print(f"Encoding Error: {e}")
     exit()
     
# 2. One-Hot Encoding
try:
    encoded_features = ohe.transform(input_data[oh_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(oh_cols))
    encoded_df.index = input_data.index 
    
    input_data_processed = pd.concat([input_data.drop(columns=oh_cols), encoded_df], axis=1)
except Exception as e:
    print(f"Encoding Error: {e}")
    exit()

# 3. Scaling
try:
    # Ensure X_columns order.
    for col in X_columns:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0 # Handle missing columns if any
            
    model_input = input_data_processed[X_columns]
    model_input_scaled = scaler.transform(model_input)
except KeyError as e:
    print(f"Feature Mismatch: {e}")
    print("Missing columns:", set(X_columns) - set(input_data_processed.columns))
    exit()

# 4. Prediction (Loop over models)
print("\n--- Predictions ---")
for name, model in models.items():
    prediction = model.predict(model_input_scaled)
    result_index = prediction[0]

    # Decode
    if len(target_classes) > 1:
        result_label = target_classes[result_index]
        result = result_label
    else:
        result = "Approved" if result_index == 1 else "Rejected"

    print(f"Model: {name}")
    print(f"Result: {result} (Raw: {result_index})")
