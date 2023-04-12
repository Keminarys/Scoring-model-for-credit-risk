from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np
import shap
import json

#Init
app = FastAPI(debug=True)

#Data 
lgbm = joblib.load('LGBM_Thresh_04785.sav')
data = joblib.load('data_api.pkl')
list_ID = data.index.tolist()

#Main
if __name__ == '__main__' : 
   uvicorn.run(app, host="127.0.0.1", port="8000")
