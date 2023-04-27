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
list_id  = re.get(API_id)

#App design
st.set_page_config(layout="wide")
st.title('Home Credit Risk Application : Does the applicant is capable of repaying a requested loan ?')  
with st.sidebar : 
  st.write("Choose an applicant ID in the dropdown list below")
  st.selectbox((list_ID))
