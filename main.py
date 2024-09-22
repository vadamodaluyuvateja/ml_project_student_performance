import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set the title of the app
st.title("Student Exam Performance Indicator")

# Create input fields for the user
gender = st.selectbox("Gender", ["Select your Gender", "male", "female"])
ethnicity = st.selectbox("Race or Ethnicity",
                         ["Select Ethnicity", "group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education",
                                           ["Select Parent Education", "associate's degree", "bachelor's degree",
                                            "high school", "master's degree", "some college", "some high school"])
lunch = st.selectbox("Lunch Type", ["Select Lunch Type", "free/reduced", "standard"])
test_preparation_course = st.selectbox("Test Preparation Course", ["Select Test Course", "none", "completed"])
reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100)

# Button for prediction
if st.button("Predict your Maths Score"):
    # Validate inputs
    if gender == "Select your Gender" or ethnicity == "Select Ethnicity" or \
       parental_level_of_education == "Select Parent Education" or lunch == "Select Lunch Type" or \
       test_preparation_course == "Select Test Course":
        st.error("Please select all fields correctly.")
    else:
        try:
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            st.success(f'The predicted score is: {results[0]}')

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
