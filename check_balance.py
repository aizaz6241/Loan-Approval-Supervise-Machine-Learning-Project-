import pandas as pd

try:
    df = pd.read_csv('loan_approval_data.csv')
    print("Class Distribution:")
    print(df['Loan_Approved'].value_counts(normalize=True))
    print("\nUnique Values:")
    print(df['Loan_Approved'].unique())
except Exception as e:
    print(e)
