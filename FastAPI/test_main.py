import requests
from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import json

data = joblib.load('data_api.pkl')
list_ID = data.index.tolist()

def test_get_home_message():
    response_message = requests.get('https://apprenticeship-credit-risk.onrender.com/home')
    assert response_message.json() == {'message' : 'Welcome to my application, the purpose of this is to predict if an applicant is capable of repaying a loan.'}

def test_get_all_id():
    response_id = requests.get('https://apprenticeship-credit-risk.onrender.com/all_applicants')
    assert response_id.json() == list_ID

def test_get_pred():
    response_pred = requests.get('https://apprenticeship-credit-risk.onrender.com/predict/261031')
    assert str(response_pred.json()) == str([0.096, 'Applicant with ID : 261031, based on the model, is not capable of repaying the loan'])  
