import streamlit as st
import pickle
import pandas as pd
from helper import preprocess

st.subheader("ThinkHumble's Customer Churn Prediction")
tabs = st.tabs(["Single Entry", "Bulk Prediction"])

# Create content for each tab
with tabs[0]:
    data = {}
    st.subheader("Please fill out the form below with the customer details you have.")
    with st.form(key='churn_form'):
        # Input fields for each feature
        data['CustomerID'] = 0
        data['Age'] = st.number_input('Age', min_value=18, max_value=80, value=40)
        data['Gender'] = st.selectbox('Gender', ['Male', 'Female'])
        data['ContractType'] = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        data['MonthlyCharges'] = st.number_input('Monthly Charges', min_value=18.0, max_value=150.0, value=65.0)
        data['TotalCharges'] = st.number_input('Total Charges', value=0.0)
        data['TechSupport'] = st.selectbox('Tech Support', ['Yes', 'No'])
        data['InternetService'] = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        data['PaperlessBilling'] = st.selectbox('Paperless Billing', ['Yes', 'No'])
        data['PaymentMethod'] = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
        data['Tenure'] = st.number_input('Tenure (months)', min_value=0, max_value=72, value=0)

        # Submit button
        submit_button = st.form_submit_button(label='Predict Churn')
        if submit_button:
            
            df = pd.DataFrame(data,index=[1])
            data = preprocess(df)

            
            with open('churn_model.pkl', 'rb') as file:
                model = pickle.load(file)
            pred = model.predict(data)
            if pred[0] == 0:
                st.success('Customer is Likely to Stay!')
            else:
                st.error('Customer is Likely to Churn!')



with tabs[1]:
    import pandas as pd

    st.header('File Upload')
    st.info('Please make sure the data is in the correct format of raw-data with column names and their order as mentioned: CustomerID, Age, Gender, ContractType, MonthlyCharges, TotalCharges, TechSupport, InternetService, PaperlessBilling, PaymentMethod, Tenure')


    # File uploader widget
    try:
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            # Determine file type and read accordingly
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            df1 = df.copy()
            data = preprocess(df)
            with open('churn_model.pkl', 'rb') as file:
                model = pickle.load(file)
            pred = model.predict(data)
            df1['Churn'] = pred
            csv = df1.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download the CSV with predictions",
                data=csv,
                file_name='data_with_predictions.csv',
                mime='text/csv'
            )
        
    except Exception as e:
        st.write(e)
        st.error('Not in the supported format')


