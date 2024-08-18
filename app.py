import streamlit as st
import pandas as pd
import tensorflow as tf

# Load the model
model_path = "ann_model.h5"
model = tf.keras.models.load_model(model_path)

# Define the Streamlit app
def main():
    st.title("Early Detection of Hepatitis B")

    # Right sidebar for code implementation and user input
    st.write("Fill in the details below:")

    age = st.number_input('Age of the Patient', min_value=0)

    sex = st.radio("Gender of the Patient", ["Male", "Female"])
    steroid = st.radio("Received steroid treatment?", ["Yes", "No"])
    antivirals = st.radio("Undergoing antiviral treatment?", ["Yes", "No"])
    fatigue = st.radio("Experiencing fatigue?", ["Yes", "No"])
    malaise = st.radio("Experiencing malaise?", ["Yes", "No"])
    anorexia = st.radio("Has anorexia?", ["Yes", "No"])
    liver_big = st.radio("Liver enlarged?", ["Yes", "No"])
    liver_firm = st.radio("Liver firm?", ["Yes", "No"])
    spleen_palpable = st.radio("Spleen palpable?", ["Yes", "No"])
    spiders = st.radio("Spider Nevi present?", ["Yes", "No"])
    ascites = st.radio("Ascites present?", ["Yes", "No"])
    varices = st.radio("Varices present?", ["Yes", "No"])
    bilirubin = st.number_input('Bilirubin levels', min_value=0.0)
    alk_phosphate = st.number_input('Alkaline Phosphate Levels', min_value=0.0)
    sgot = st.number_input('SGOT levels', min_value=0.0)
    albumin = st.number_input("Albumin Levels", min_value=0.0)
    protime = st.number_input('Prothrombin Time', min_value=0.0)
    histology = st.radio("Histological Examination done?", ["Yes", "No"])

    # Left sidebar for symptoms
    with st.sidebar:
        st.header("Symptoms of Hepatitis B")

        st.markdown("""
            <p style='color:red; font-size:20px;'>Anorexia:</p> Restrictive eating patterns, excessive exercise, weight loss or failure to gain weight, hair loss, cold intolerance, fatigue, osteoporosis, menstrual irregularities
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Fatigue:</p> Persistent tiredness or exhaustion, lack of energy
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Malaise:</p> General discomfort, unease, lack of energy or motivation
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Liver Enlargement (Hepatomegaly):</p> Abdominal pain or fullness, jaundice, nausea, vomiting, fatigue
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Liver Firmness:</p> Firmness upon palpation of the liver, abdominal discomfort
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Palpable Spleen (Splenomegaly):</p> Abdominal pain, early satiety, anemia
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Spider Nevi (Spider Angiomas):</p> Small, red, spider-like blood vessels on the skin, commonly on the face, neck, and chest
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Ascites:</p> Abdominal swelling due to fluid buildup, discomfort, shortness of breath, weight gain
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Varices:</p> Enlarged veins in the esophagus or stomach, potential bleeding, vomiting blood, black stools
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Elevated Bilirubin Levels (Hyperbilirubinemia):</p> Jaundice, yellowing of the skin and eyes, dark urine, fatigue
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Elevated Alkaline Phosphatase Levels:</p> Jaundice, itching, bone pain, fatigue
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Elevated SGOT Levels (Aspartate Aminotransferase):</p> Liver damage symptoms, fatigue, weakness, jaundice
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Low Albumin Levels:</p> Swelling (edema), fatigue, weakness, ascites
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style='color:red; font-size:20px;'>Prolonged Prothrombin Time:</p> Easy bruising, excessive bleeding, increased risk of hemorrhage
        """, unsafe_allow_html=True)

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

    # Convert categorical inputs to numerical values
    for key, value in input_data.items():
        if value == "Male":
            input_data[key] = 1
        elif value == "Female":
            input_data[key] = 0
        elif value == "Yes":
            input_data[key] = 1
        elif value == "No":
            input_data[key] = 0

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Define the prediction process in Streamlit
    if st.button('Predict'):
        try:
            # Make predictions using the model
            predictions = model.predict(input_df)
            
            # Assuming the model outputs a single probability value for binary classification
            prediction_score = predictions[0][0]
            
            # Determine the class label based on a threshold
            threshold = 0.5
            prediction_label = 1 if prediction_score > threshold else 0
            
            # Display prediction results
            st.subheader("Prediction Result")
            
            # Format the prediction score for display
            chance = f"{prediction_score * 100:.2f}"
            
            # Provide feedback based on the prediction label
            if prediction_label == 0:
                st.error(f"The model indicates a potential Hepatitis B infection with a {100-float(chance)}% probability. Consult a healthcare professional for further evaluation.")
            else:
                st.success(f"The model suggests a low likelihood of Hepatitis B with a {chance}% probability. Continue regular check-ups and consult a healthcare professional if needed.")
            
            # Display the prediction score
            st.write(f"Prediction Score: {prediction_score:.4f}")
        
        except KeyError as e:
            st.error(f"Error: {e}. Check input data columns and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
