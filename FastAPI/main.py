from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np
import shap
import json

#Init
app = FastAPI()

#Data 
lgbm = joblib.load('LGBM_Thresh_04785.sav')
data = joblib.load('data_api.pkl')
# with open('data_api.pkl', 'rb') as f:
#       data = pd.read_pickle(f)
list_ID = data.index.tolist()

#Function
@app.get("/home")
async def home():
    return {'message' : 'Welcome to my application, the purpose of this is to predict if an applicant is capable of repaying a loan.'}

@app.get("/all_applicants")
async def all_applicants():
      return list_ID

@app.get("/data_id/{applicant_id}")
async def data_id(applicant_id : int):
    data_applicant = data[data.index == applicant_id]
    data_applicant = data_applicant.to_json(orient='records', lines=True)
    return data_applicant
    
@app.get("/predict/{applicant_id}")
async def predict(applicant_id : int):
   predictions = lgbm.predict_proba(data).tolist()
   predict_proba = []
   threshold = 0.4785  
   for pred, ID in zip(predictions, list_ID):
      if ID == applicant_id:
         predict_proba.append(pred[1])
         if predict_proba[0] >= threshold :
                result_message = f'Applicant with ID : {ID}, based on the model, is capable of repaying the loan'
         else : result_message = f'Applicant with ID : {ID}, based on the model, is not capable of repaying the loan'
   return round(predict_proba[0],3), result_message

# @app.get('/ID_explainer/{applicant_id}')
# async def ID_explainer(applicant_id : int):
#    index_ID = []
#    for ind, ID in enumerate(list_ID):
#         if list_ID[ind] == applicant_id:
#             index_ID.append(ind)
#             shap_values_applicant = shap_values[index_ID[0]][0]
#             json_shap_applicant = json.dumps(shap_values_applicant.tolist())
#    return {'shap_client':json_shap_applicant}

#Main
if __name__ == '__main__' : 
   uvicorn.run(app, host="0.0.0.0", port="10000") #Deployed on render
   #uvicorn.run(app, host='127.0.0.1', port='8000') Localhost
