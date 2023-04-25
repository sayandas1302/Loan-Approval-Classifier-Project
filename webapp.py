# importing libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# calling the saved scaler
with open("./scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# calling the saved model
with open('./svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# function for preprocessing the input
def preprocess(d):
    num_var = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    d1 = d.copy()

    # scalling the numerical variables
    d1[num_var] = pd.DataFrame(scaler.transform(d1[num_var]))
    
    # encoding the categorical variables
    d1['Credit_History'] = d1['Credit_History'].map({'Yes': 1, 'No': 0})

    d1['Gender_Male'] = d1['Gender'].map({'Male': 1, 'Female': 0})
    d1 = d1.drop(['Gender'], axis = 1)

    d1['Married_Yes'] = d1['Married'].map({'Yes': 1, 'No': 0})
    d1 = d1.drop(['Married'], axis = 1)

    d1['Dependents_1'] = d1['Dependents'].map({'0': 0, '1': 1, '2': 0, '3+': 0})
    d1['Dependents_2'] = d1['Dependents'].map({'0': 0, '1': 0, '2': 1, '3+': 0})
    d1['Dependents_3+'] = d1['Dependents'].map({'0': 0, '1': 0, '2': 0, '3+': 1})
    d1 = d1.drop(['Dependents'], axis = 1)

    d1['Education_Not_Graduate'] = d1['Education'].map({"Graduate": 0, "Not-Graduate": 1})
    d1 = d1.drop(['Education'], axis = 1)

    d1['Self_Employed_Yes'] = d1['Self_Employed'].map({"Yes": 1, "No": 0})
    d1 = d1.drop('Self_Employed', axis = 1)

    d1['Property_Area_Semiurban'] = d1['Property_Area'].map({'Semi-Urban': 1, 'Urban': 0, 'Rural': 0})
    d1['Property_Area_Urban'] = d1['Property_Area'].map({'Semi-Urban': 0, 'Urban': 1, 'Rural': 0})
    d1 = d1.drop(['Property_Area'], axis = 1)

    return d1

def guess():
    # processing the inserted data
    df = pd.DataFrame({
        "Gender": [], "Married": [], "Dependents": [], "Education": [],
        "Self_Employed": [], "ApplicantIncome": [], "CoapplicantIncome": [], "LoanAmount": [],
        "Loan_Amount_Term": [], "Credit_History": [], "Property_Area": []
    })
    
    df.loc[0] = [
        gen, mar_stat, depen, edu,
        self_emp, income, co_income, loan_amt,
        loan_amt_term,credit_his, prop_area
    ]
    df = preprocess(df)
    
    # predicting the outcome
    y_pred = model.predict(df)[0]

    # confidence/probability in favour of out count
    prop = model.predict_proba(df)[0][y_pred]
    
    # showing the result
    if y_pred == 1:
        st.markdown("##### Your loan will be approved.") 
    else:
        st.markdown("##### Your loan will not be approved.")

    st.markdown(f"##### Confidence: {round(100*prop, 2)}%")

# setup of the GUI 
st.markdown(f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/stack-black-money-coin-banking-currency-business-finance-cash-dollar-treasure-earnings-financial-profit-market-investment-stock-exchange-dark-3d-background-with-success-economy-income_79161-2032.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """, unsafe_allow_html=True)

st.title("Check the Applicability of Your Loan")
st.markdown("##### Input the following information about you to know whether your loan will approved")

with st.container():
    st.header("Inputs")
    
    column1, column2, column3 = st.columns(3)

    with column1:
        gen = st.radio("Gender", ("Male", "Female"), index = 0, horizontal = True)
        mar_stat = st.radio("Marital Status", ("Yes", "No"), index = 0, horizontal = True)
        depen = st.radio("No of Dependents", ("0", "1", "2", "3+"), index = 0, horizontal = True)
        edu = st.radio("Education", ("Graduate", "Not-Graduate"), index = 0, horizontal = True)
   
    with column2:
        self_emp = st.radio("Self Employed", ("Yes", "No"), index = 0, horizontal = True)
        credit_his = st.radio("Credit History", ("Yes", "No"), index = 0, horizontal = True)
        prop_area = st.radio("Property Area", ("Urban", "Rural", "Semi-Urban"), index = 0, horizontal = True)

    with column3:
        income = st.slider("Income", min_value = 0, max_value = 100000)
        co_income = st.slider("Co-applicant Income", min_value = 0, max_value = 100000)
        loan_amt = st.slider("Loan Amount", min_value = 0, max_value = 100000)
        loan_amt_term = st.slider("Loan Amount Term", min_value = 0, max_value = 100000)

    but_pred = st.button("Predict")

with st.container():
    st.header("Results")
    
    if but_pred:
       guess()