# app.py

import streamlit as st
import joblib
import numpy as np

# Load the trained stacked model
model = joblib.load('best_stacking_model.pkl')

# Streamlit app title and description
st.title("Stacked Model Prediction App")
st.write("This app allows you to input features and get predictions from the stacked model.")

# Input fields for features
st.sidebar.header("Input Features")
num_features = 10  # Adjust this to the number of features your model expects
input_features = []

for i in range(num_features):
    value = st.sidebar.number_input(f"Feature {i+1}", value=0.0)
    input_features.append(value)

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Convert input to numpy array
        features = np.array(input_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Display result
        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
