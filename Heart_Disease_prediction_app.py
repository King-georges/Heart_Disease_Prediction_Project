import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Heart_Disease_Prediction.csv')

st.title('Heart Disease Prediction App')
st.write("""
# Heart Disease Prediction App

This app predicts the likelihood of having Heart Disease

Data obtained from the University of California Irvine data repository and is used to predict heart disease
 https://data.world/informatics-edu/heart-disease-prediction/workspace/file?filename=+Heart_Disease_Prediction.csv
""")