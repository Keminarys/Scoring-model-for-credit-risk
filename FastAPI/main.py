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
#data = joblib.load('data_api.pkl')
with open('data_api.pkl', 'rb') as f:
      data = pd.read_pickle(f)
#shap_values = joblib.load('shap_values_data.pkl')
with open('shap_values_data.pkl', 'rb') as f:
      shap_values = pd.read_pickle(f)
list_ID = data.index.tolist()

#Function
@app.post('/')
def get():
    return {'message' : 'Welcome to my application, the purpose of this is to predict if an applicant is capable of repaying a loan.'}

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
   return predict_proba[0], result_message

@app.get('/ID_explainer/{applicant_id}')
async def ID_explainer(applicant_id : int):
   index_ID = []
   for ind, ID in enumerate(list_ID):
        if list_ID[ind] == applicant_id:
            index_ID.append(ind)
            shap_values_id = shap_values_data[index_ID][0]
            json_shap_id = json.dumps(shap_values_id.tolist())
   return {'shap_id':json_shap_id}

@app.get('/model_explainer')
async def model_explainer():
    shap_values_model = [value.tolist() for value in shap_values]
    json_shap_model = json.dumps(shap_values_model)
    return {'shap_values_model':json_shap_model}      
         
#Main
if __name__ == '__main__' : 
   uvicorn.run(app, host="0.0.0.0", port="10000")
