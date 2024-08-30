import pandas as pd
import numpy as np
import pickle
def preprocess(test_df):
    # Ensure correct data types for categorical columns
    categorical_cols = ['Gender', 'ContractType', 'TechSupport', 'InternetService', 'PaperlessBilling', 'PaymentMethod']
    for col in categorical_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype('category')
    
    # Define mappings
    gender_mapping = {'Male': 0, 'Female': 1}
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    tech_support_mapping = {'Yes': 1, 'No': 0}
    internet_service_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    paperless_billing_mapping = {'Yes': 1, 'No': 0}
    payment_method_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3}
    churn_mapping = {'Yes': 1, 'No': 0}

    # Apply mappings
    if 'Gender' in test_df.columns:
        test_df['Gender'] = test_df['Gender'].map(gender_mapping)
    if 'ContractType' in test_df.columns:
        test_df['ContractType'] = test_df['ContractType'].map(contract_mapping)
    if 'TechSupport' in test_df.columns:
        test_df['TechSupport'] = test_df['TechSupport'].map(tech_support_mapping)
    if 'InternetService' in test_df.columns:
        test_df['InternetService'] = test_df['InternetService'].map(internet_service_mapping)
    if 'PaperlessBilling' in test_df.columns:
        test_df['PaperlessBilling'] = test_df['PaperlessBilling'].map(paperless_billing_mapping)
    if 'PaymentMethod' in test_df.columns:
        test_df['PaymentMethod'] = test_df['PaymentMethod'].map(payment_method_mapping)
    if 'Churn' in test_df.columns:
        test_df['Churn'] = test_df['Churn'].map(churn_mapping)

    # Fill missing values
    if 'TotalCharges' in test_df.columns:
        test_df['TotalCharges'] = test_df['TotalCharges'].fillna(0)  # Assuming 0 is a valid fill for missing TotalCharges

    # Calculate derived features
    if 'TotalCharges' in test_df.columns and 'Tenure' in test_df.columns:
        test_df['AverageMonthlyCharges'] = test_df['TotalCharges'] / np.where(test_df['Tenure'] == 0, 1, test_df['Tenure'])
        test_df['CustomerLifetimeValue'] = test_df['MonthlyCharges'] * test_df['Tenure']
    else:
        raise ValueError("Columns 'TotalCharges' or 'Tenure' are missing in the DataFrame")

    # Create TenureGroup
    if 'Tenure' in test_df.columns:
        test_df['TenureGroup'] = pd.cut(test_df['Tenure'], bins=[0, 12, 24, np.inf], labels=['New', 'Intermediate', 'Long-term'])
    
    # Calculate TotalChargesPerMonth
    if 'TotalCharges' in test_df.columns and 'Tenure' in test_df.columns:
        test_df['TotalChargesPerMonth'] = test_df['TotalCharges'] / np.where(test_df['Tenure'] > 0, test_df['Tenure'], 1)
    
    # Categorize PaymentMethod into Electronic and Non-Electronic
    if 'PaymentMethod' in test_df.columns:
        test_df['Electronic'] = test_df['PaymentMethod'].apply(lambda x: 1 if x in [0, 2] else 0)
    
    # Interaction Terms
    if 'MonthlyCharges' in test_df.columns and 'Tenure' in test_df.columns:
        test_df['MonthlyCharges_Tenure'] = test_df['MonthlyCharges'] * test_df['Tenure']
    if 'InternetService' in test_df.columns and 'TechSupport' in test_df.columns:
        test_df['Internet_TechSupport'] = ((test_df['InternetService'] != 2) & (test_df['TechSupport'] != 0)).astype(int)
    
    # Log Transformation of Total Charges
    if 'TotalCharges' in test_df.columns:
        test_df['LogTotalCharges'] = np.log(test_df['TotalCharges'].replace(0, np.nan))  # Handle zero values to avoid log(0)
    
    with open('default_values.pkl', 'rb') as file:
        default_values = pickle.load(file)

    
    for column, values in default_values.items():
        if column in test_df.columns:
            imputation_value = values['imputation_value']
            if test_df[column].dtype == 'object':
                # For categorical columns
                test_df[column].fillna(imputation_value, inplace=True)
            else:
                # For numerical columns
                test_df[column].fillna(imputation_value, inplace=True)
    low_corr_features = ['Age', 'PaymentMethod', 'Gender', 'InternetService', 'TenureGroup', 'Tenure', 'Electronic']
    df_reduced = test_df.drop(columns=low_corr_features)
    return df_reduced