import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# Load the pre-trained credit card model
try:
    with open('card_model.pkl', 'rb') as file:
        model = pickle.load(file)
    model_status = "Model Loaded Successfully"
except Exception as e:
    model_status = f"Failed to load model: {e}"

# Add custom CSS to make the page more glamorous
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        color: #4B4B4B;
        font-family: 'Arial', sans-serif;
    }
    h2 {
        color: #4B9CD3;
        font-size: 36px;
        font-weight: bold;
    }
    .stButton button {
        background-color: #4B9CD3;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #357ABD;
    }
    .stRadio div, .stSelectbox div {
        font-size: 18px;
    }
    .status-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        font-weight: bold;
        color: #2C6E49;
    }
    .input-section {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .image-section {
        text-align: center;
        margin-top: 20px;
    }
    .image-section img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Display model status
st.write(f'<div class="status-indicator">Status: {model_status}</div>', unsafe_allow_html=True)

# Display header and images
st.write("## ✨ Credit Card Eligibility Checker ✨")
st.write("Fill out the form below and press **'Check My Eligibility'** to determine if you qualify for a credit card.")


# Define mappings for categorical variables
income_type_map = {0: "Working", 1: "Commercial associate", 2: "Pensioner", 3: "State servant", 4: "Unemployed"}
education_type_map = {0: "Higher education", 1: "Secondary / secondary special", 2: "Incomplete higher", 3: "Lower secondary", 4: "Academic degree"}
family_status_map = {0: "Married", 1: "Single / not married", 2: "Civil marriage", 3: "Separated", 4: "Widow"}
housing_type_map = {0: "House / apartment", 1: "With parents", 2: "Municipal apartment", 3: "Rented apartment", 4: "Office apartment", 5: "Co-op apartment"}
occupation_type_map = {0: "Laborers", 1: "Core staff", 2: "Managers", 3: "Drivers", 4: "High skill tech staff", 5: "Accountants"}

# Function to collect user input
def get_user_input():
    st.write('<div class="input-section">', unsafe_allow_html=True)
    st.write("### Enter Your Details")

    # Collect user input through Streamlit widgets
    CODE_GENDER = st.radio("Select your gender:", ('Male', 'Female'))
    FLAG_OWN_CAR = st.radio("Do you own a car?", ('Yes', 'No'))
    FLAG_OWN_REALTY = st.radio("Do you own real estate?", ('Yes', 'No'))
    CNT_CHILDREN = st.number_input("Number of children:", min_value=0, step=1, help="Enter the total number of children in your family.")
    AMT_INCOME_TOTAL = st.number_input("Annual income (in currency):", min_value=0.0, format="%.2f", help="Enter your total annual income.")
    
    # Dropdowns instead of numeric codes
    NAME_INCOME_TYPE = st.selectbox("Income type:", list(income_type_map.values()))
    NAME_EDUCATION_TYPE = st.selectbox("Education level:", list(education_type_map.values()))
    NAME_FAMILY_STATUS = st.selectbox("Marital status:", list(family_status_map.values()))
    NAME_HOUSING_TYPE = st.selectbox("Housing type:", list(housing_type_map.values()))
    DAYS_BIRTH = st.number_input("Days since birth (positive value):", min_value=0, max_value=365*120, help="Enter the number of days since your birth (positive value).")
    DAYS_EMPLOYED = st.number_input("Days since employment (positive value):", min_value=0, max_value=365*50, help="Enter the number of days since you started employment (positive value).")
    FLAG_MOBIL = st.radio("Do you have a mobile phone?", ('Yes', 'No'))
    FLAG_WORK_PHONE = st.radio("Do you have a work phone?", ('Yes', 'No'))
    FLAG_PHONE = st.radio("Do you have a phone?", ('Yes', 'No'))
    FLAG_EMAIL = st.radio("Do you have an email?", ('Yes', 'No'))
    OCCUPATION_TYPE = st.selectbox("Occupation type:", list(occupation_type_map.values()))
    CNT_FAM_MEMBERS = st.number_input("Number of family members:", min_value=0.0, format="%.1f", help="Enter the total number of family members.")

    st.write('</div>', unsafe_allow_html=True)

    # Convert categorical inputs to numeric
    input_data = {
        'CODE_GENDER': 1 if CODE_GENDER == 'Male' else 0,
        'FLAG_OWN_CAR': 1 if FLAG_OWN_CAR == 'Yes' else 0,
        'FLAG_OWN_REALTY': 1 if FLAG_OWN_REALTY == 'Yes' else 0,
        'CNT_CHILDREN': CNT_CHILDREN,
        'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'NAME_INCOME_TYPE': list(income_type_map.keys())[list(income_type_map.values()).index(NAME_INCOME_TYPE)],
        'NAME_EDUCATION_TYPE': list(education_type_map.keys())[list(education_type_map.values()).index(NAME_EDUCATION_TYPE)],
        'NAME_FAMILY_STATUS': list(family_status_map.keys())[list(family_status_map.values()).index(NAME_FAMILY_STATUS)],
        'NAME_HOUSING_TYPE': list(housing_type_map.keys())[list(housing_type_map.values()).index(NAME_HOUSING_TYPE)],
        'DAYS_BIRTH': DAYS_BIRTH,
        'DAYS_EMPLOYED': DAYS_EMPLOYED,
        'FLAG_MOBIL': 1 if FLAG_MOBIL == 'Yes' else 0,
        'FLAG_WORK_PHONE': 1 if FLAG_WORK_PHONE == 'Yes' else 0,
        'FLAG_PHONE': 1 if FLAG_PHONE == 'Yes' else 0,
        'FLAG_EMAIL': 1 if FLAG_EMAIL == 'Yes' else 0,
        'OCCUPATION_TYPE': list(occupation_type_map.keys())[list(occupation_type_map.values()).index(OCCUPATION_TYPE)],
        'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS
    }

    # Convert input data to DataFrame
    return pd.DataFrame([input_data])

# Collect user input
input_df = get_user_input()

# Display the input data
st.write("### Your Input Data:")
st.dataframe(input_df)  # Show the input data

# Button to trigger prediction
if st.button('Check My Eligibility'):
    if model_status == "Model Loaded Successfully":
        # Predict using the trained model
        try:
            prediction = model.predict(input_df)[0]

            # Output result
            if prediction == 0:
                st.success("🎉 Congratulations! You are eligible for a credit card.")
            else:
                st.error("😢 Sorry, you are not eligible for a credit card at this time.")
        except Exception as e:
            st.error(f"An error occurred")
