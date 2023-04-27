import pandas as pd
import numpy as np
import shap
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests as re

#Fixed Variable
API_id = "https://apprenticeship-credit-risk.onrender.com/all_applicants/"
API_pred = "https://apprenticeship-credit-risk.onrender.com/predict/"
API_data = "https://apprenticeship-credit-risk.onrender.com/data_id/"
expected_value = -0.457
#explainer = joblib.load("https://github.com/Keminarys/Scoring-model-for-credit-risk/blob/main/FastAPI/explainer_lgbm.pkl")
#shap_global = joblib.load("https://github.com/Keminarys/Scoring-model-for-credit-risk/blob/main/FastAPI/shap_values_global.pkl")
#shap_graph = joblib.load("https://github.com/Keminarys/Scoring-model-for-credit-risk/blob/main/FastAPI/shap_values_graphs.pkl")
list_ID  = re.get(API_id).json()
ask = ["No","Yes"]

#App design
st.set_page_config(layout="wide")
st.title('Home Credit Risk Application : Does the applicant is capable of repaying a requested loan ?')  
with st.sidebar : 
  st.divider()
  st.write("ID SELECTION")
  applicant_selected = st.selectbox("Choose an applicant ID in the dropdown list below", list_ID)
  st.divider()
  st.write("MODEL INTERPRETABILITY")
  choice_glob = st.radio("Would you like to see global model interpretability graph ?", ask)
  choice_loc = st.radio("Would you like to see local model interpretability graph ?", ask)
  st.divider()
  col_plot = st.write("2-Dimensional Plots from SHAP")
  st.divider()
  choice_df = st.radio("Would you like to see dataframe ?", ask)

json_applicant = re.get(API_data+str(applicant_selected)).json()
json_applicant = json.loads(json_applicant)
#data_applicant =  pd.DataFrame.from_dict()
pred_applicant = re.get(API_pred+str(applicant_selected)).json()

with st.container():
  if choice_df == 'Yes' :
    st.write(json_applicant)
  
