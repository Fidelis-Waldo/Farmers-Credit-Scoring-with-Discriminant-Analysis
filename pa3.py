import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('discriminant_analysis_model.pkl')

# Streamlit app title
st.title("Discriminant Analysis Classifier")

# User input form
st.header("Input Your Data")

# 1. Age (years)
age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)

# 2. Gender
gender = st.radio("Gender", options=["Male", "Female"])
gender_value = 0 if gender == "Male" else 1  # Convert to numerical value

# 3. Education (years)
education = st.number_input("Education (years)", min_value=0, max_value=16, step=1)

# Ensure education is less than age
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
                "This individual is **<span style='color:red;'>NOT CREDIT ELIGIBLE</span>**.",
                unsafe_allow_html=True
            )
        else:
                st.markdown(
                "This individual is **<span style='color:green;'>CREDIT ELIGIBLE</span>**.",
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
