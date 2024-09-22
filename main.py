import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set the title and a header for the app
st.title("Student Exam Performance Indicator")
st.header("Predict Your Math Score")

# Add background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('YOUR_IMAGE_URL') no-repeat center center fixed;
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background for sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a sidebar for user inputs
st.sidebar.header("Input Fields")

# Create input fields for the user
gender = st.sidebar.selectbox("Gender", ["Select your Gender", "male", "female"])
ethnicity = st.sidebar.selectbox("Race or Ethnicity",
                                  ["Select Ethnicity", "group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.sidebar.selectbox("Parental Level of Education",
                                                   ["Select Parent Education", "associate's degree", "bachelor's degree",
                                                    "high school", "master's degree", "some college", "some high school"])
lunch = st.sidebar.selectbox("Lunch Type", ["Select Lunch Type", "free/reduced", "standard"])
test_preparation_course = st.sidebar.selectbox("Test Preparation Course", ["Select Test Course", "none", "completed"])
reading_score = st.sidebar.number_input("Reading Score out of 100", min_value=0, max_value=100)
writing_score = st.sidebar.number_input("Writing Score out of 100", min_value=0, max_value=100)

# Button for prediction
if st.sidebar.button("Predict your Maths Score"):
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

# Footer with your email
st.markdown("---")
st.markdown("For questions or feedback, contact me at:(mailto:tejavadamodula@gmail.com)")
