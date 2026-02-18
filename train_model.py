import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
try:
    df = pd.read_csv("loan_approval_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: loan_approval_data.csv not found.")
    exit()

# --- Preprocessing ---

# Drop Applicant_ID if it exists
if "Applicant_ID" in df.columns:
    df = df.drop("Applicant_ID", axis=1)

# Handle Missing Values
# Drop rows where target 'Loan_Approved' is missing INITIALIZATION
if "Loan_Approved" in df.columns:
    df = df.dropna(subset=["Loan_Approved"])

categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

# Drop target from numerical_cols if present (it might not be yet if it's object)
if "Loan_Approved" in numerical_cols:
    numerical_cols = numerical_cols.drop("Loan_Approved")

num_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
# Ensure we don't impute the target variable if it's in categorical_cols (it is)
feature_cat_cols = [col for col in categorical_cols if col != "Loan_Approved"]
df[feature_cat_cols] = cat_imputer.fit_transform(df[feature_cat_cols])

# Encode Target Variable
le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])
print(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Encode Education_Level (Ordinal)
le_education = LabelEncoder()
if "Education_Level" in df.columns:
    df["Education_Level"] = le_education.fit_transform(df["Education_Level"])

# One-Hot Encode other categoricals
# Note: Education_Level is already encoded, Loan_Approved is target.
oh_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
# Verify cols exist
oh_cols = [col for col in oh_cols if col in df.columns]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_data = ohe.fit_transform(df[oh_cols])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(oh_cols), index=df.index)

# Combine features
df_processed = pd.concat([df.drop(columns=oh_cols), encoded_df], axis=1)

# Separate Features and Target
X = df_processed.drop("Loan_Approved", axis=1)
y = df_processed["Loan_Approved"]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

import traceback

# --- Model Selection ---

models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}

print(f"Target classes: {le_target.classes_}")
print(f"y shape: {y.shape}")
print(f"y unique values: {np.unique(y)}")

print("\nModel Evaluation:")
try:
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Determine average type
        average_type = 'macro' if len(le_target.classes_) > 2 else 'binary'
        pos_label = 1 if len(le_target.classes_) == 2 else None

        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average='macro') # macro for imbalance check
        if average_type == 'binary':
             recall_approved = recall_score(y_test, y_pred, pos_label=1)
        else:
             recall_approved = recall_score(y_test, y_pred, average='macro')
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print(f"Recall (Approved): {recall_approved:.4f}")
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in le_target.classes_]))
        
        # Store trained model
        trained_models[name] = model

except Exception:
    print("An error occurred during training/evaluation:")
    print(traceback.format_exc())
    exit(1)

# --- Save Artifacts ---

artifacts = {
    'models': trained_models, # Save dictionary of all models
    'num_imputer': num_imputer,
    'cat_imputer': cat_imputer,
    'le_education': le_education,
    'ohe': ohe,
    'scaler': scaler,
    'numerical_cols': numerical_cols,
    'feature_cat_cols': feature_cat_cols,
    'oh_cols': oh_cols,
    'X_columns': X.columns,
    'target_classes': le_target.classes_
}

joblib.dump(artifacts, 'loan_model.pkl')
print(f"All models and artifacts saved to loan_model.pkl")
