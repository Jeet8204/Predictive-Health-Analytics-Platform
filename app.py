
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np 

# --- Page Configuration ---
st.set_page_config(
    page_title="Health Assistant AI 🧑‍⚕️",
    layout="wide",
    page_icon="🏥"
)


st.markdown(
    """
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        color: #007bff; /* Primary blue color */
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 1.8em;
        color: #333333;
        text-align: center;
        margin-bottom: 1.5em;
    }
    /* stTextInput and stNumberInput styles are fine */
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
        color: #333333;
    }
    .stNumberInput>div>div>input {
        background-color: #f0f2f6;
        color: #333333;
    }
    .stButton>button {
        background-color: #28a745; /* Green for success */
        color: white;
        font-weight: bold;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
    }
    /* This class is for general styled text, not for st.success/st.error directly */
    .prediction-result-text { 
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 1em;
    }
    .sidebar .sidebar-content {
        background-color: #e0f7fa; /* Light blue for sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Path and Model Loading ---
working_dir = os.path.dirname(os.path.abspath(__file__))


models_dir = os.path.join(working_dir, 'saved_models')
if not os.path.exists(models_dir):
    st.error(f"Error: 'saved_models' directory not found at {models_dir}. Please ensure your model files are correctly placed.")
    st.stop() 

try:
    diabetes_model = pickle.load(open(os.path.join(models_dir, 'diabetes_model.sav'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(models_dir, 'heart_disease_model.sav'), 'rb'))
    parkinsons_model = pickle.load(open(os.path.join(models_dir, 'parkinsons_model.sav'), 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}. Please ensure all model files are in the 'saved_models' directory.")
    st.stop()


# --- Sidebar for Navigation ---
with st.sidebar:
    st.markdown("## 🩺 Navigation")
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )
    st.markdown("---")
    st.info("Navigate through different prediction models using the menu above.")

# --- Main Content Area ---

# --- Diabetes Prediction Page ---
if selected == 'Diabetes Prediction':
    st.markdown("<h1 class='main-header'>🩸 Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enter the patient's details to predict diabetes risk.</p>", unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17, value=1, help='Number of times pregnant')
        with col2:
            Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120, help='Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
        with col3:
            BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=122, value=70, help='Diastolic blood pressure')
        with col1:
            SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=99, value=20, help='Triceps skin fold thickness')
        with col2:
            Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, value=79, help='2-Hour serum insulin')
        with col3:
            BMI = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=67.1, value=25.0, help='Body Mass Index')
        with col1:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.372, format="%.3f", help='A function which scores likelihood of diabetes based on family history')
        with col2:
            Age = st.number_input('Age of the Person (years)', min_value=1, max_value=120, value=30)

    diab_diagnosis = ''

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button('Predict Diabetes Risk'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            
            
            user_input_as_np_array = np.asarray(user_input, dtype=np.float64)

            diab_prediction = diabetes_model.predict(user_input_as_np_array.reshape(1, -1))

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is predicted to be **Diabetic** 😔'
                st.error(diab_diagnosis) 
            else:
                diab_diagnosis = 'The person is predicted to be **Non-Diabetic**! 🎉'
                st.success(diab_diagnosis) 
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please check your input values.")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Heart Disease Prediction Page ---
if selected == 'Heart Disease Prediction':
    st.markdown("<h1 class='main-header'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Input patient's cardiovascular parameters.</p>", unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=50)
        with col2:
            sex = st.number_input('Sex (0 = Female, 1 = Male)', min_value=0, max_value=1, value=1, help='0: Female, 1: Male')
        with col3:
            cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, value=0, help='0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic')
        with col1:
            trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=90, max_value=200, value=120)
        with col2:
            chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=100, max_value=600, value=200)
        with col3:
            fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)', min_value=0, max_value=1, value=0, help='1 if FBS > 120 mg/dl, 0 otherwise')
        with col1:
            restecg = st.number_input('Resting Electrocardiographic Results (0-2)', min_value=0, max_value=2, value=0, help='0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy')
        with col2:
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
        with col3:
            exang = st.number_input('Exercise Induced Angina (0 = No, 1 = Yes)', min_value=0, max_value=1, value=0, help='1: Yes, 0: No')
        with col1:
            oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=1.0, format="%.1f")
        with col2:
            slope = st.number_input('Slope of the Peak Exercise ST Segment (0-2)', min_value=0, max_value=2, value=1, help='0: Upsloping, 1: Flat, 2: Downsloping')
        with col3:
            ca = st.number_input('Major Vessels Colored by Flourosopy (0-4)', min_value=0, max_value=4, value=0, help='Number of major vessels (0-3) colored by flourosopy, 4: unknown')
        with col1:
            thal = st.number_input('Thalassemia (0-2)', min_value=0, max_value=2, value=1, help='0: Normal, 1: Fixed defect, 2: Reversable defect')

    heart_diagnosis = ''

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button('Predict Heart Disease Risk'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input_as_np_array = np.asarray(user_input, dtype=np.float64)

            heart_prediction = heart_disease_model.predict(user_input_as_np_array.reshape(1, -1))

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is predicted to have **Heart Disease** 💔'
                st.error(heart_diagnosis) 
            else:
                heart_diagnosis = 'The person is predicted to be **Healthy (No Heart Disease)**! 💪'
                st.success(heart_diagnosis)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please check your input values.")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Parkinson's Prediction Page ---
if selected == "Parkinsons Prediction":
    st.markdown("<h1 class='main-header'>🚶 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Provide voice feature data to assess Parkinson's risk.</p>", unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.number_input('MDVP:Fo(Hz) - Average vocal fundamental frequency', min_value=50.0, max_value=250.0, value=150.0, format="%.3f")
        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz) - Maximum vocal fundamental frequency', min_value=50.0, max_value=350.0, value=200.0, format="%.3f")
        with col3:
            flo = st.number_input('MDVP:Flo(Hz) - Minimum vocal fundamental frequency', min_value=50.0, max_value=250.0, value=100.0, format="%.3f")
        with col4:
            Jitter_percent = st.number_input('MDVP:Jitter(%) - Jitter (percent)', min_value=0.0000, max_value=0.0120, value=0.0050, format="%.4f")
        with col5:
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs) - Jitter (absolute)', min_value=0.00000, max_value=0.00012, value=0.00005, format="%.5f")
        with col1:
            RAP = st.number_input('MDVP:RAP - Relative Amplitude Perturbation', min_value=0.0000, max_value=0.0070, value=0.0025, format="%.4f")
        with col2:
            PPQ = st.number_input('MDVP:PPQ - Five-point Period Perturbation Quotient', min_value=0.0000, max_value=0.0070, value=0.0025, format="%.4f")
        with col3:
            DDP = st.number_input('Jitter:DDP - Average absolute difference of differences of periods', min_value=0.0000, max_value=0.0210, value=0.0075, format="%.4f")
        with col4:
            Shimmer = st.number_input('MDVP:Shimmer - Shimmer', min_value=0.0000, max_value=0.0900, value=0.0400, format="%.4f")
        with col5:
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB) - Shimmer (dB)', min_value=0.000, max_value=1.500, value=0.500, format="%.3f")
        with col1:
            APQ3 = st.number_input('Shimmer:APQ3 - Three-point Amplitude Perturbation Quotient', min_value=0.0000, max_value=0.0500, value=0.0200, format="%.4f")
        with col2:
            APQ5 = st.number_input('Shimmer:APQ5 - Five-point Amplitude Perturbation Quotient', min_value=0.0000, max_value=0.0700, value=0.0300, format="%.4f")
        with col3:
            APQ = st.number_input('MDVP:APQ - Amplitude Perturbation Quotient', min_value=0.0000, max_value=0.1500, value=0.0400, format="%.4f")
        with col4:
            DDA = st.number_input('Shimmer:DDA - Average absolute differences between amplitudes of consecutive periods', min_value=0.0000, max_value=0.1500, value=0.0600, format="%.4f")
        with col5:
            NHR = st.number_input('NHR - Noise-to-Harmonics Ratio', min_value=0.0000, max_value=0.5000, value=0.2000, format="%.4f")
        with col1:
            HNR = st.number_input('HNR - Harmonics-to-Noise Ratio', min_value=0.000, max_value=35.000, value=20.000, format="%.3f")
        with col2:
            RPDE = st.number_input('RPDE - Recurrence Period Density Entropy', min_value=0.000, max_value=0.800, value=0.500, format="%.3f")
        with col3:
            DFA = st.number_input('DFA - Detrended Fluctuation Analysis', min_value=0.000, max_value=1.000, value=0.600, format="%.3f")
        with col4:
            spread1 = st.number_input('spread1 - Nonlinear dynamical complexity measure 1', min_value=-8.000, max_value=-2.000, value=-4.000, format="%.3f")
        with col5:
            spread2 = st.number_input('spread2 - Nonlinear dynamical complexity measure 2', min_value=0.000, max_value=0.500, value=0.200, format="%.3f")
        with col1:
            D2 = st.number_input('D2 - Nonlinear dynamical complexity measure 3', min_value=1.000, max_value=4.000, value=2.000, format="%.3f")
        with col2:
            PPE = st.number_input('PPE - Pitch Period Entropy', min_value=0.000, max_value=0.700, value=0.200, format="%.3f")

    parkinsons_diagnosis = ''

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("Predict Parkinson's Risk"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            
            user_input_as_np_array = np.asarray(user_input, dtype=np.float64)

            parkinsons_prediction = parkinsons_model.predict(user_input_as_np_array.reshape(1, -1))

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person is predicted to have **Parkinson's Disease** 😔"
                st.error(parkinsons_diagnosis) 
            else:
                parkinsons_diagnosis = "The person is predicted to be **Healthy (No Parkinson's Disease)**! 🎉"
                st.success(parkinsons_diagnosis) 
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please check your input values.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 2em; font-size: 0.9em; color: #666;'>
        <p><strong>Disclaimer:</strong> This application is for informational and educational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
    </div>
    """,
    unsafe_allow_html=True
)
