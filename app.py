import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from pycaret.classification import *



#  Loading the model back to a Python Environment¶
model = load_model("smote_model")

# Define the Streamlit app
def main():
    st.title("Early Detection of Hepatitis B")

    # Define the input fields for the features
    age = st.number_input('Age of the Patient')
    sex = st.selectbox("Gender of the Patient", ["Male", "Female"])
    steroid = st.selectbox("Have the patient received steroids treatment (yes/no)", ["Yes", "No"])
    antivirals = st.selectbox("Is the patient undergoing antiviral treatment (yes/no)", ["Yes", "No"])
    fatigue = st.selectbox("Does the patient experience fatigue (yes/no)", ["Yes", "No"])
    malaise = st.selectbox("Does the patient experience malaise (yes/no)", ["Yes", "No"])
    anorexia = st.selectbox("Does the patient have anorexia (yes/no)", ["Yes", "No"])
    liver_big= st.selectbox("Is the patient liver enlarged (yes/no)", ["Yes", "No"])
    liver_firm = st.selectbox("Is the patient liver firm (yes/no)", ["Yes", "No"])
    spleen_palpable = st.selectbox("Is the patient spleen palpable (yes/no)", ["Yes", "No"])
    spiders = st.selectbox("Presence of Spiders Nevi (yes/no)", ["Yes", "No"])
    ascites = st.selectbox("Presence of Ascites (yes/no)", ["Yes", "No"])
    varices = st.selectbox("Presence of Varices (yes/no)", ["Yes", "No"])
    bilirubin = st.number_input('Bilirubin levels')
    alk_phosphate = st.number_input('Alkaline Phosphate Levels')
    sgot = st.number_input('Serum Glutamic Oxaloacetic Transaminase (SGOT) levels')
    albumin = st.number_input("Albumin Levels")
    protime = st.number_input('Protime Time')
    histology = st.selectbox("Have the patient undergone Histological Examination (yes/no)", ["Yes", "No"])
    # class_ = st.selectbox("Status", ["Yes", "No"])


    # Create a dictionary from the input fields
    input_data = {
        'age': age,
        'sex': sex,
        'steroid': steroid,
        'antivirals': antivirals,
        'fatigue': fatigue,
        'malaise': malaise,
        'anorexia': anorexia,
        'liver_big': liver_big,
        'liver_firm': liver_firm,
        'spleen_palpable': spleen_palpable,
        'spiders': spiders,
        'ascites': ascites,
        'varices': varices,
        'bilirubin': bilirubin,
        'alk_phosphate': alk_phosphate,
        'sgot': sgot,
        'albumin': albumin,
        'protime': protime,
        'histology': histology,
    }

    for key, value in input_data.items():
        if value == "Male":
            input_data[key] = 1
        if value == "Female":
            input_data[key] = 0
        if value == "Yes":
            input_data[key] = 1
        if value == "No":
            input_data[key] = 0


    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        try:

            # Make predictions using the model
            prediction_result = predict_model(model, data=input_df)
            st.write(prediction_result)

            # Extract specific details from the prediction result
            prediction_label = prediction_result['prediction_label'][0]
            prediction_score = prediction_result['prediction_score'][0]

            print(type(prediction_label))

            # Display specific details
            st.subheader("Prediction Result")

            if prediction_label == 0:  # Assuming 0 indicates a negative result
                chance = str(prediction_score*100)[:4]
                st.error(f"""The model indicates that you may have Hepatitis B, with a {chance}% probability of being affected by the infection. 
                        It is crucial to seek further evaluation and confirmation from a healthcare professional and to follow their recommended course of action""")

            else: # Assuming 1 indicates a positive result
                chance = str(prediction_score*100)[:4]
                st.success(f"""
                            The model suggests that you are unlikely to have Hepatitis B, with a {chance}% probability of being free from the infection. Nonetheless, 
                        it is important to continue regular check-ups and consult a healthcare professional if you have any concerns.
                            """)
            st.write(prediction_score)

        except KeyError as e:
            st.error(f"Error: {e}. Check input data columns and try again.")

if __name__ == '__main__':
    main()
