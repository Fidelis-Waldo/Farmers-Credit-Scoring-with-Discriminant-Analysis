import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('discriminant_analysis_model.pkl')

# Streamlit app title
st.title("Discriminant Analysis Classifier")

# User input form
st.header("Input Your Data")

# 1. Age (years) - Categories as Dropdown
st.subheader("Age Group")
age_group = st.selectbox(
    "Select your age group:",
    options=[
        "18 - 24 years (Young Adults / Early Borrowers)",
        "25 - 34 years (Early Career / Growing Professionals)",
        "35 - 44 years (Established Professionals / Family Builders)",
        "45 - 54 years (Mid Career / Peak Earners)",
        "55 - 64 years (Pre-Retirement / Financial Planners)",
        "65+ years (Retirees / Seniors)"
    ]
)

# Assign numerical values to age categories
age_mapping = {
    "18 - 24 years (Young Adults / Early Borrowers)": 20,
    "25 - 34 years (Early Career / Growing Professionals)": 30,
    "35 - 44 years (Established Professionals / Family Builders)": 40,
    "45 - 54 years (Mid Career / Peak Earners)": 50,
    "55 - 64 years (Pre-Retirement / Financial Planners)": 60,
    "65+ years (Retirees / Seniors)": 65
}
age = age_mapping[age_group]  # Numerical value for the selected category

# 2. Gender
gender = st.radio("Gender", options=["Male", "Female"])
gender_value = 0 if gender == "Male" else 1  # Convert to numerical value

# 3. Education (years) - Categories as Dropdown
st.subheader("Education Level")
education_level = st.selectbox(
    "Select your highest education level:",
    options=[
        "No Education",
        "Primary Education",
        "Secondary Education",
        "Tertiary Education"
    ]
)

# Assign numerical values to education levels
education_mapping = {
    "No Education": 0,
    "Primary Education": 6,
    "Secondary Education": 12,
    "Tertiary Education": 16
}
education = education_mapping[education_level]  # Numerical value for the selected category

# Validation: Ensure education is less than age
if education >= age:
    st.warning("Education value must be less than Age. Please adjust your input.")

# 4. Farming Experience (years)
farming_experience = st.number_input("Farming Experience (years)", min_value=0, step=1)

# Ensure farming experience is less than age by at least 15 years
if farming_experience >= age - 15:
    st.warning("Farming Experience must be at least 15 years less than Age. Please adjust your input.")

# 5. Distance to Loan Source (km)
distance_to_loan_source = st.number_input("Distance to Loan Source (km)", min_value=0.0, step=0.1)

# 6. Loan to Asset Ratio
st.subheader("Loan to Asset Ratio")
loan_value = st.number_input("Loan Value", min_value=0.0, step=0.1)
asset_value = st.number_input("Asset Value", min_value=0.1, step=0.1)  # Ensure non-zero
loan_to_asset_ratio = loan_value / asset_value if asset_value > 0 else 0.0

# 7. Operating Expenditure to Income Ratio
st.subheader("Operating Expenditure to Income Ratio")
operating_expenditure = st.number_input("Operating Expenditure", min_value=0.0, step=0.1)
income_value = st.number_input("Income", min_value=0.1, step=0.1)  # Ensure non-zero
opex_to_income_ratio = operating_expenditure / income_value if income_value > 0 else 0.0

# 8. Outstanding Loan to Asset Ratio
st.subheader("Outstanding Loan to Asset Ratio")
outstanding_loan = st.number_input("Outstanding Loan Value", min_value=0.0, step=0.1)
outstanding_loan_to_asset_ratio = outstanding_loan / asset_value if asset_value > 0 else 0.0

# 9. Farm Size
farm_size = st.number_input("Farm Size (acres)", min_value=0.0, step=0.1)

# Validation to ensure all required fields are valid
if st.button("Classify"):
    if education >= age:
        st.error("Please ensure Education value is less than Age.")
    elif farming_experience >= age - 15:
        st.error("Please ensure Farming Experience is at least 15 years less than Age.")
    else:
        # Prepare input data for prediction
        input_data = np.array([[
            age, gender_value, education, farming_experience,
            distance_to_loan_source, loan_to_asset_ratio,
            opex_to_income_ratio, outstanding_loan_to_asset_ratio,
            farm_size
        ]])

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.markdown(
                "This individual is **<span style='color:red;'>Not Credit Eligible</span>** for this loan amount.",
                unsafe_allow_html=True
            )

            # Calculate the maximum loan value to make the individual credit eligible
            min_loan_value = loan_value
            step = 100  # Decrement step for loan value
            max_iterations = 10000  # Limit iterations to avoid infinite loops
            iteration = 0

            while iteration < max_iterations and min_loan_value > 0:
                min_loan_value -= step
                new_loan_to_asset_ratio = min_loan_value / asset_value if asset_value > 0 else 0.0
                updated_input_data = input_data.copy()
                updated_input_data[0, 5] = new_loan_to_asset_ratio
                new_prediction = model.predict(updated_input_data)

                if new_prediction[0] == 1:  # If the individual becomes credit eligible
                    break

                iteration += 1

            if min_loan_value > 0:
                st.write(
                    f"The maximum loan value that would make this individual credit eligible is: "
                    f"**{min_loan_value:.2f}**."
                )
            else:
                st.write(
                    "It is not possible to make this individual credit eligible by reducing the loan value."
                )
        else:
            st.markdown(
                "This individual is **<span style='color:green;'>Credit Eligible</span>**.",
                unsafe_allow_html=True
            )

        # Display probabilities
        st.subheader("Prediction Probabilities")
        pay_back_probability = prediction_proba[0][1] * 100
        default_probability = prediction_proba[0][0] * 100
        st.write(
            f"The probability of the individual paying back is {pay_back_probability:.2f}% "
            f"while the probability of the individual defaulting is {default_probability:.2f}%."
        )
