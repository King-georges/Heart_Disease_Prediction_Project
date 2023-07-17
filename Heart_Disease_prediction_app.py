import streamlit as st
import requests
import plotly.express as px
from streamlit_lottie import st_lottie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Heart_Disease_Prediction.csv')

st.set_page_config(page_title="Heart Disease Prediction App",
                page_icon= ":bar_chart:")
st.title('Heart Disease Prediction app')
st.write("""
## This app predicts the likelihood of having Heart Disease

Data obtained from the University of California Irvine data repository and is used to predict heart disease
 https://data.world/informatics-edu/heart-disease-prediction/workspace/file?filename=+Heart_Disease_Prediction.csv
""")
st.subheader('Training Data')
st.write(df.head())
st.write(df.describe())

st.subheader('Visualisation')
st.write("Bar chart:")
st._legacy_bar_chart(df)

x = df.drop(columns = 'Heart Disease')

y = df['Heart Disease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

st.sidebar.header('User Input Features')
def user_features():
        Thallium = st.sidebar.slider('Thallium', 3, 7)
        Numberofvesselsfluro = st.sidebar.slider('Number of vessels fluro', 0, 3)
        Exerciseangina = st.sidebar.slider('Exercise angina', 0, 1)
        Maxheartrate = st.sidebar.slider('Max HR,', 71, 202)
        STdepression = st.sidebar.slider('ST Depression', 0, 6.2)
        Chestpaintype = st.sidebar.slider('Chest pain type', 1, 4)
        SlopeofST = st.sidebar.slider('Slope of ST', 1, 3)
        Age = st.sidebar.slider('Age', 29, 77 )
