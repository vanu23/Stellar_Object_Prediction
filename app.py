import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the model
model = load("xg_model.joblib")

# Load the dataset and fit scaler
stellar = pd.read_csv("star_classification.csv")

if "obj_ID" in stellar.columns.values:
    stellar.drop(["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID","MJD","fiber_ID"], axis=1, inplace=True)

# Drop the target variable
X = stellar.drop(['class'], axis=1)
scaler = StandardScaler()
scaler.fit(X)

st.header("Stellar Object Classification")
st.header("Enter Object Information")

# Input fields for each feature
input_data = []
for col in X.columns:
    value = st.number_input(f"{col}", value=float(X[col].mean()))
    input_data.append(value)

input_data = np.array([input_data])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Star Class"):
    prediction = model.predict(input_data_scaled)
    class_map = {0: "Class Galaxy", 1: "Class Quasar", 2: "Class Star"}  # Update with actual class mappings
    predicted_class = class_map[prediction[0]]
    st.success(f"The predicted class is: {predicted_class}")

#https://stellarobjectpredictionvanshika.streamlit.app/
#GALAXY- 1.23766E+18,135.6891066,32.49463184,23.87882,22.2753,20.39501,19.16573,18.79371,3606,301,2,79,6.54378E+18,GALAXY,0.6347936,5812,56354,171
#QSO- 1.23766E+18,145.8830055,47.30048358,21.73992,21.53095,21.26763,21.36257,21.15861,2821,301,2,33,8.22824E+18,QSO,2.07568,7308,56709,596
#STAR- 1.23768E+18,14.38313522,3.214326196,21.82154,20.5573,19.94918,19.76057,19.55514,7712,301,5,425,9.85507E+18,STAR,-0.000440276,8753,57373,258