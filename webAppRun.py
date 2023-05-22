from flask import Flask, render_template, request
import pandas as pd
import pickle

# calling the saved scaler
with open("./scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# calling the saved model
with open('./svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

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

def guess(df):
    # processing the inserted data
    df = preprocess(df)
    
    # predicting the outcome
    y_pred = model.predict(df)[0]

    # confidence/probability in favour of out count
    prop = model.predict_proba(df)[0][y_pred]
    
    # showing the result
    if y_pred == 1:
        message = "Your loan will be approved."
    else:
        message = "Your loan will not be approved."

    conf = round(100*prop, 2)

    return (message, conf)

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    message = "No values inserted yet !"
    conf = 0
    
    if request.method == 'POST':
        gen = request.form.get('gender')
        mar_stat = request.form.get('mstatus')
        depen = request.form.get('ndep')
        prop_area = request.form.get('parea')
        edu = request.form.get('edu')
        self_emp = request.form.get('semp')
        credit_his = request.form.get('chistory')
        income = request.form.get('income')
        co_income = request.form.get('coincome')
        loan_amt = request.form.get('amount')
        loan_amt_term = request.form.get('term')

        df = pd.DataFrame({
            "Gender": [], "Married": [], "Dependents": [], "Education": [],
            "Self_Employed": [], "ApplicantIncome": [], "CoapplicantIncome": [], "LoanAmount": [],
            "Loan_Amount_Term": [], "Credit_History": [], "Property_Area": []
        })

        if None in [gen, mar_stat, depen, edu,
            self_emp, income, co_income, loan_amt,
            loan_amt_term,credit_his, prop_area]:
            message = "Not all the inputs are provided !"
        else:
            df.loc[0] = [
                gen, mar_stat, depen, edu,
                self_emp, income, co_income, loan_amt,
                loan_amt_term,credit_his, prop_area]

            message, conf = guess(df)
    
    return render_template('webapphome.html', message=message, conf=conf)

app.run(debug=True)