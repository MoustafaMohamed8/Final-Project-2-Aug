import streamlit as st
import joblib
import numpy as np
from utils import process_new

## Load the model

model=joblib.load('svc_model.pkl')


def Heartattack_classification():
    ## Title
    st.title('Heart Attack Risk Prediciton')
    st.markdown('---')

    ## input fields
    
    ## input fields
    age=st.number_input('How old are you?',step=1)
    sex=st.selectbox('What is your gender? 0=Male,1=Female',options=[0,1])
    cp=st.selectbox('What is your chest pain type? 0=Typical Angina , 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic',options=[0,1,2,3])
    trtbps=st.number_input('What is your Resting Blood Pressure?',step=1)
    chol=st.number_input('What is your Cholestrol Level',step=1)
    fbs=st.selectbox('What is your fasting blood sugar level? 0=fbs<120,1= fbs>120',options=[0,1])
    restecg=st.selectbox('What is your resting electrocardiographic result? 0=Normal,1=ST-T Wavem, 2=Left Ventrical Hypertrophy',options=[0,1,2])
    thalachh=st.number_input('What is your Max HeartRate Acheived?',step=1)
    exng=st.selectbox('Did you Ever experience Exercise Induced Angina? 0=No,1=Yes',options=[0,1])
    oldpeak=st.number_input('what is your previous peak?',step=0.1)
    slp=st.selectbox('Slope of the peak exercise st segment',options=[0,1,2])
    caa=st.selectbox('No. of Major vessels',options=[0,1,2,3,4])
    thall=st.selectbox('Thalium Stress test results',options=[0,1,2,3])
    age_category=st.selectbox('What is your age category?',options=['Middle-Aged','Senior','Young'])
    st.markdown('---')

    if st.button('Predict Whether you are at a risk of getting a heart attack or Not.'):
        new_data=np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,
                          oldpeak,slp,caa,thall,age_category])

        X_processed=process_new(x_new=new_data)

    ## Predict
        y_pred=model.predict(X_processed)
        if y_pred == 1:
           y_pred ='Very High'
        else:
            y_pred='Low'
    ## Display
        st.success(f'Your Risk of getting a heart attack is {y_pred} ')
    
    return None



if __name__=='__main__':
    Heartattack_classification()

