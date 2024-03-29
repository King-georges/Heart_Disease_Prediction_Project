import streamlit as st
import requests
from streamlit_lottie import st_lottie
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Heart_Disease_Prediction.csv')



st.set_page_config(page_title="Heart Disease Prediction App",
                page_icon= ":bar_chart:")
st.title('Heart Disease Prediction app')
st.write("""
## This app predicts the likelihood of having Heart Disease.

Data obtained from the University of California Irvine data repository and is used to predict heart disease. Patients were classified as having or not having heart disease based on cardiac catheterization, the gold standard. If they had more than 50% narrowing of a coronary artery they were labeled as having heart disease.

 https://data.world/informatics-edu/heart-disease-prediction/workspace/file?filename=+Heart_Disease_Prediction.csv . 
         It is most suitable for medical experts with the necessary domain knowledge on Heart disease
""")
st.write('---')
def load_lottieurl(url):
     r = requests.get(url)
     if r.status_code != 200:
          return None
     return r.json()

lottie_coding = "https://lottie.host/1ea84b8b-b12a-466e-b4b5-8889e2af00de/FqIw2IixwF.json"
x = df.drop(columns = 'Heart Disease')

y = df['Heart Disease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

left_column, right_column = st.columns(2)
with left_column:
     st.subheader('Training Data')
     st.write(x_train.describe())
with right_column:
     st_lottie(lottie_coding, height=00, key="heart")

st.subheader('Visualisation')
st.write("Bar chart:")
st._legacy_bar_chart(x_train)

st.write("1 represents Male and 0 represents Female")



st.sidebar.header('User Input Features')
def user_features():
        thallium = st.sidebar.slider('Thallium', 0, 7)
        numberofvesselsfluro = st.sidebar.slider('Number of vessels fluro', 0, 3)
        bloodpressure = st.sidebar.slider('BP', 70, 300)
        fbsover120 = st.sidebar.slider('FBS over 120', 0, 1)
        cholesterol = st.sidebar.slider('Cholesterol', 100, 600)
        ekgresults = st.sidebar.slider('EKG results', 0, 2)
        exerciseangina = st.sidebar.slider('Exercise angina', 0, 1)
        maxheartrate = st.sidebar.slider('Max HR', 60, 210)
        stdepression = st.sidebar.slider('ST Depression', 0, 6)
        chestpaintype = st.sidebar.slider('Chest pain type', 1, 4)
        slopeofST = st.sidebar.slider('Slope of ST', 1, 3)
        age = st.sidebar.slider('Age', 18, 77 )
        sex = st.selectbox('Sex', [1, 0])
        

    
        user_features = {
               'Thallium': thallium,
               'Number of vessels fluro': numberofvesselsfluro,
               'BP' : bloodpressure,
               'FBS over 120' : fbsover120,
               'Cholesterol' : cholesterol,
               'EKG results' : ekgresults,
               'Exercise angina' : exerciseangina,
               'Max HR' : maxheartrate,
               'ST Depression' : stdepression,
               'Chest pain type' : chestpaintype,
               'Slope of ST' : slopeofST,
               'Age' : age,
               'Sex' : sex

    } 
        report_data = pd.DataFrame(user_features, index=[0])
        return report_data

user_data = user_features()

scaler = StandardScaler()
user_data_scaled = scaler.fit_transform(user_data)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

reg = LogisticRegression()
reg.fit(x_train, y_train)

st.subheader('Accuracy')
st.write(str(accuracy_score(y_test, reg.predict(x_test))*100)+'%')

user_result = reg.predict(user_data_scaled)
st.subheader('Your Report: ')

if user_result[0] == 1:
    output = 'You are likely to Test Positive For Heart Disease'
else:
    output = 'You are likely to Test Negative For Heart Disease'

st.write(output)

def local_css(file_name):
     with open(file_name) as f:
          st.markdown (f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("Style/style.css")

with st.container():
     st.write("---")
     st.header("Get in touch with me!")
     st.write("##")
     contact_form = '''
     <input type="hidden" name="_captcha" value="false">
     <form action="https://formsubmit.co/opondigeorge@gmail.com" method="POST">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="Message" placeholder="Your message here" required></textarea>
     <button type="submit">Send</button>
</form>
'''
lottie_animation = "https://lottie.host/87b1eda6-f65b-47f0-b06a-fa93ee6f73ba/bNjIHSLbzV.json"
left_column, right_column = st.columns (2)
with left_column:
     st.markdown (contact_form, unsafe_allow_html=True)
     with right_column:
          st_lottie(lottie_animation, height=00, key="message")
          
