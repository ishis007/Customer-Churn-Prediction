#scaler is exported as sxcaler.pkl
#model is exported as model.pkl
import streamlit as st
import joblib
import numpy as np


scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("please enter the values and hit the predict button for getting prediction")

st.divider()

age=st.number_input("Enter age",min_value=10,max_value=130,value=30)

tenure=st.number_input("Enter Tenure",min_value=0,max_value=130, value=10)

Monthlycharge=st.number_input("Enter Monthly Charge",min_value=30,max_value=300)

gender=st.selectbox("Enter Gender",["Male","Female"])

st.divider()

predictbutton=st.button("Predict!")

if predictbutton:

    gender_selected=1 if gender=="Female" else 0

    x=[age,gender_selected,tenure,Monthlycharge]

    x1=np.array(x)

    x_array=scaler.transform([x1])

    prediction=model.predict(x_array)[0]

    predicted="churn" if prediction==1 else "not churn"

    st.write(f"Predicted:{predicted}")
    
else:
    st.write("Please enter the values and use predict button")
