import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Medical Diagnostic System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------

# Dictionary mapping technical feature names to human-readable labels and descriptions
FEATURE_CONFIG = {
    # --- HEART ---
    "age": {"label": "Age", "help": "Age of the patient in years."},
    "sex": {"label": "Sex", "help": "1 = Male, 0 = Female"},
    "cp": {"label": "Chest Pain Type", "help": "0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"},
    "trestbps": {"label": "Resting Blood Pressure", "help": "Resting blood pressure (in mm Hg on admission to the hospital). Normal is < 120/80."},
    "chol": {"label": "Serum Cholesterol", "help": "Serum cholesterol in mg/dl."},
    "fbs": {"label": "Fasting Blood Sugar", "help": "1 = > 120 mg/dl, 0 = False. Indicates diabetes risk."},
    "restecg": {"label": "Resting ECG Results", "help": "0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy"},
    "thalach": {"label": "Max Heart Rate", "help": "Maximum heart rate achieved during test."},
    "exang": {"label": "Exercise Induced Angina", "help": "1 = Yes, 0 = No"},
    "oldpeak": {"label": "ST Depression", "help": "ST depression induced by exercise relative to rest."},
    "slope": {"label": "Slope of ST Segment", "help": "0: Upsloping, 1: Flat, 2: Downsloping"},
    "ca": {"label": "Number of Major Vessels", "help": "Number of major vessels (0-3) colored by flourosopy."},
    "thal": {"label": "Thal", "help": "0: Normal, 1: Fixed Defect, 2: Reversable Defect"},

    # --- DIABETES ---
    "Pregnancies": {"label": "Pregnancy Ratio", "help": "Number of times pregnant."},
    "Glucose": {"label": "Glucose Level", "help": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test."},
    "BloodPressure": {"label": "Blood Pressure", "help": "Diastolic blood pressure (mm Hg)."},
    "SkinThickness": {"label": "Skin Thickness", "help": "Triceps skin fold thickness (mm)."},
    "Insulin": {"label": "Insulin Level", "help": "2-Hour serum insulin (mu U/ml)."},
    "BMI": {"label": "BMI", "help": "Body mass index (weight in kg / (height in m)^2)."},
    "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree", "help": "Diabetes pedigree function (genetic score)."},
    "Age": {"label": "Age", "help": "Age of the patient."},

    # --- BREAST CANCER (Sample of main features) ---
    "radius_mean": {"label": "Mean Radius", "help": "Mean of distances from center to points on the perimeter."},
    "texture_mean": {"label": "Mean Texture", "help": "Standard deviation of gray-scale values."},
    "perimeter_mean": {"label": "Mean Perimeter", "help": "Mean size of the core tumor."},
    "area_mean": {"label": "Mean Area", "help": "Mean area of the core tumor."},
    "smoothness_mean": {"label": "Mean Smoothness", "help": "Mean of local variation in radius lengths."},

    # --- HEPATITIS ---
    "Steroid": {"label": "Steroids", "help": "1 = No, 2 = Yes"},
    "Antivirals": {"label": "Antivirals", "help": "1 = No, 2 = Yes"},
    "Fatigue": {"label": "Fatigue", "help": "1 = No, 2 = Yes"},
    "Malaise": {"label": "Malaise", "help": "1 = No, 2 = Yes"},
    "Anorexia": {"label": "Anorexia", "help": "1 = No, 2 = Yes"},
    "Liver Big": {"label": "Liver Big", "help": "1 = No, 2 = Yes"},
    "Bilirubin": {"label": "Bilirubin", "help": "Bilirubin level."},
    "Albumin": {"label": "Albumin", "help": "Albumin level."},
    
    # --- PARKINSONS (Sample) ---
    "MDVP:Fo(Hz)": {"label": "Fund. Freq. (Fo)", "help": "Average vocal fundamental frequency"},
    "MDVP:Fhi(Hz)": {"label": "Max Freq. (Fhi)", "help": "Maximum vocal fundamental frequency"},
    "MDVP:Flo(Hz)": {"label": "Min Freq. (Flo)", "help": "Minimum vocal fundamental frequency"},
    "MDVP:Jitter(%)": {"label": "Jitter (%)", "help": "Measure of variation in fundamental frequency"},
}

def get_feature_info(name):
    """Retrieve label and help text, falling back to capitalized name."""
    clean_name = name.strip()
    if clean_name in FEATURE_CONFIG:
        return FEATURE_CONFIG[clean_name]
    
    # Fallback/Generic Logic
    return {
        "label": clean_name.replace("_", " ").title(), 
        "help": f"Enter value for {clean_name}."
    }

# -----------------------------------------------------------------------------
# APP LOGIC
# -----------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('models/multi_disease_pipeline.pkl')
    except Exception as e:
        return None

pipeline = load_pipeline()

if pipeline is None:
    st.error("System Error: Diagnostic models could not be loaded. Please ensure the pipeline file exists.")
    st.stop()

models = pipeline['models']
scalers = pipeline['scalers']
feature_names = pipeline['feature_names']
feature_stats = pipeline.get('feature_stats', {})
accuracies = pipeline.get('accuracies', {})

# Sidebar Navigation
with st.sidebar:
    st.title("MedDiagnostic System")
    st.markdown("---")
    
    st.subheader("Configuration")
    
    # Disease Selector
    display_map = {k: k.replace('_', ' ').title() for k in models.keys()}
    inv_display_map = {v: k for k, v in display_map.items()}
    
    selected_display_name = st.selectbox(
        "Select Medical Condition", 
        list(display_map.values())
    )
    selected_disease = inv_display_map[selected_display_name]
    
    st.markdown("---")
    
    # Model Performance Metric in Sidebar
    st.subheader("Model Reliability")
    acc = accuracies.get(selected_disease, 0.0)
    st.metric(label="Validation Accuracy", value=f"{acc:.1%}")
    st.caption("Accuracy metric based on clinical test datasets.")
    
    st.markdown("---")
    st.markdown("**Version:** 2.1.0 Pro")

# Main Content Area
st.title(f"{selected_display_name} Assessment")
st.info(f"Please input the clinical parameters below. Hover over the '?' icon next to each field for guidance.")

# Main Form
with st.form("diagnostic_form"):
    st.subheader("Patient Clinical Data")
    
    features = feature_names.get(selected_disease, [])
    stats = feature_stats.get(selected_disease, {})
    
    # Organize inputs into 3 columns
    cols = st.columns(3)
    input_values = {}
    
    for i, feat in enumerate(features):
        col_idx = i % 3
        col = cols[col_idx]
        
        # Get defaults and metadata
        feat_stat = stats.get(feat, {})
        min_curr = feat_stat.get('min', 0.0)
        max_curr = feat_stat.get('max', 10.0)
        mean_curr = feat_stat.get('mean', 5.0)
        
        # Buffer for input range
        value_range = max_curr - min_curr
        if value_range == 0: value_range = 10.0
        
        min_input = min_curr - (value_range * 0.2)
        max_input = max_curr + (value_range * 0.2)
        
        # Get human readable info
        info = get_feature_info(feat)
        
        with col:
            val = st.number_input(
                label=info['label'],
                min_value=float(min_input),
                max_value=float(max_input),
                value=float(mean_curr),
                format="%.2f",
                help=info['help']
            )
            input_values[feat] = val
            
    st.markdown("---")
    submit_btn = st.form_submit_button("Run Diagnostic Assessment", type="primary", use_container_width=True)

# Analysis Logic & Results
if submit_btn:
    try:
        # Prepare Input
        input_list = [input_values[f] for f in features]
        input_vector = pd.DataFrame([input_list], columns=features)
        
        # Scaling
        if selected_disease in scalers:
            input_scaled = scalers[selected_disease].transform(input_vector)
        else:
            input_scaled = input_vector
            
        # Prediction
        model = models[selected_disease]
        prediction = model.predict(input_scaled)[0]
        
        # Probability
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0, 1]
        
        # ---------------------------------------------------------------------
        # RESULTS DISPLAY
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Diagnostic Results")
        
        # 1. Summary of Inputs (Engagement Feature)
        with st.expander("View Patient Input Summary", expanded=False):
            summary_data = {
                get_feature_info(k)['label']: f"{v:.2f}" 
                for k, v in input_values.items()
            }
            st.dataframe(pd.DataFrame(list(summary_data.items()), columns=["Parameter", "Value"]), use_container_width=True)

        # 2. Main Result
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK DETECTED")
                st.markdown(f"The analysis indicates a high probability of **{selected_display_name}** based on the provided clinical profiles.")
                st.markdown("**Recommendation:** Immediate clinical consultation is advised.")
            else:
                st.success("✅ LOW RISK / NEGATIVE")
                st.markdown(f"The analysis indicates a low probability of **{selected_display_name}**.")
                st.markdown("**Recommendation:** Routine monitoring as per standard protocols.")
                
        with res_col2:
            st.metric("Risk Probability", f"{probability:.1%}")
            
            # Custom progress bar color based on risk
            bar_color = ":red[" if probability > 0.5 else ":green["
            st.progress(probability)
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
