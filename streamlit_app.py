import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('Salary Category Prediction App')
st.write('Enter the details below to predict the salary category (Low, Medium, High).')

# Load the trained model
try:
    with open('naive_bayes_salary_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    st.success('Model loaded successfully!')
except FileNotFoundError:
    st.error("Error: 'naive_bayes_salary_classifier.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the LabelEncoders dictionary
try:
    with open('label_encoders.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)
    st.success('LabelEncoders loaded successfully!')
except FileNotFoundError:
    st.error("Error: 'label_encoders.pkl' not found. Please ensure the encoders file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading LabelEncoders: {e}")
    st.stop()

# Define the order of columns as expected by the model (excluding 'Salary')
# Based on the X dataframe's columns when it was created
model_features = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']

# Input fields for prediction

# Numerical input
rating = st.sidebar.slider('Rating', 1.0, 5.0, 3.5, 0.1)
salaries_reported = st.sidebar.number_input('Salaries Reported', min_value=1, value=1)

# Categorical inputs - use the loaded LabelEncoders to get options
company_name_options = list(label_encoders['Company Name'].classes_)
company_name = st.selectbox('Company Name', company_name_options)

job_title_options = list(label_encoders['Job Title'].classes_)
job_title = st.selectbox('Job Title', job_title_options)

location_options = list(label_encoders['Location'].classes_)
location = st.selectbox('Location', location_options)

employment_status_options = list(label_encoders['Employment Status'].classes_)
employment_status = st.selectbox('Employment Status', employment_status_options)

job_roles_options = list(label_encoders['Job Roles'].classes_)
job_roles = st.selectbox('Job Roles', job_roles_options)


if st.button('Predict Salary Category'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Rating': [rating],
        'Company Name': [company_name],
        'Job Title': [job_title],
        'Salaries Reported': [salaries_reported],
        'Location': [location],
        'Employment Status': [employment_status],
        'Job Roles': [job_roles]
    })

    # Apply Label Encoding to categorical features in input_data
    for col in categorical_cols:
        if col in input_data.columns:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except ValueError as ve:
                st.warning(f"Warning: '{input_data[col].iloc[0]}' not seen in training for '{col}'. This might affect prediction accuracy. Error: {ve}")
                # Handle unseen labels by assigning a default or the mode, or a new category if the model supports it.
                # For now, let's just use the known classes.
                # A more robust solution might involve adding unseen labels to the encoder or using OneHotEncoding.
                # For simplicity in this demo, we'll proceed if possible or show warning.
                # If the transform fails, it will raise a ValueError, which is caught.
                st.stop()

    # Ensure the order of columns matches the training data
    input_data = input_data[model_features]

    # Make prediction
    prediction = model.predict(input_data)

    st.subheader('Prediction:')
    st.write(f'The predicted salary category is: **{prediction[0]}**')


st.markdown("""
--- 
This app uses a Gaussian Naive Bayes classifier trained on the provided dataset."
""")
