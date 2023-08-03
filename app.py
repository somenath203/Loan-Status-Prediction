from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle


app = FastAPI()


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = pickle.load(open('loan_status_model.pkl', 'rb'))


class LoanModel(BaseModel):
    genderNo: int
    marriedNo: int
    dependentsNo: int
    educationNo: int
    selfEmployedNo: int
    applicantIncomeNo: float
    coApplicantIncomeNo: float
    loanAmountNo: float
    loanAmountTermNo: float
    creditHistoryNo: float
    propertyAreaNo: float


@app.get('/')
def welcome():
    return {
        'success': True,
        'message': 'server of loan status prediction is up and running successfully'
    }


@app.post('/predict-loan-status')
async def loan_status(loanInputFromFrontend: LoanModel):
    gender = loanInputFromFrontend.genderNo
    married = loanInputFromFrontend.marriedNo
    dependent = loanInputFromFrontend.dependentsNo
    education = loanInputFromFrontend.educationNo
    selfEmployed = loanInputFromFrontend.selfEmployedNo
    applicantIncome = loanInputFromFrontend.applicantIncomeNo
    coApplicantIncome = loanInputFromFrontend.coApplicantIncomeNo
    loanAmount = loanInputFromFrontend.loanAmountNo
    loanAmountTerm = loanInputFromFrontend.loanAmountTermNo
    creditHistory = loanInputFromFrontend.creditHistoryNo
    propertyArea = loanInputFromFrontend.propertyAreaNo


    prediction_data = pd.DataFrame([[gender, married, dependent, education, selfEmployed, applicantIncome, coApplicantIncome, loanAmount, loanAmountTerm, creditHistory, propertyArea]],
                                   columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    
    
    prediction = model.predict(prediction_data)

    pred_result_value = int(prediction[0])

    return {
        'status': 'success',
        'pred_message': pred_result_value
    }
