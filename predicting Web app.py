# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:42:20 2024

@author: avign
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/avign/Desktop/Machine Learning Model Deployment/trained_model.sav', 'rb'))


#Creating Function For Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    #giving The title 
    st.title('Diabetes Prediction')
    
    #Getting The input same as dataset
    
    Pregnancies = st.text_input('Number Of Pregnanices')
    
    Gulcose = st.text_input('Gulcose Level')
    
    BloodPressure = st.text_input('BP Value')
    
    SkinThickness =st.text_input('Skin Thickness level')
    
    Insulin = st.text_input('Insulin Level')
    
    BMI = st.text_input('BMI Value')
    
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    
    Age = st.text_input('Age')
    
    #Code For Prediction
    
    diagnosis = " "
    
    #Create a button for predict
    
    if st.button('Diabetes Result'):
        diagnosis = diabetes_prediction([Pregnancies,Gulcose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__== '__main__':
    main()
    
    
    
    
    
    