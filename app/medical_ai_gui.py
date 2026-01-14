import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Medical Diagnostic System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# -----------------------------------------------------------------------------
# KNOWLEDGE BASE (SYMPTOM MATCHING)
# -----------------------------------------------------------------------------

SYMPTOM_DB = {
    "heart": {
        "keywords": ["chest pain", "tight chest", "heart hurts", "angina", "short of breath", "dizzy", "fainting", "fatigue", "palpitations", "racing heart", "pressure in chest", "squeezing"],
        "msg": "I understand. Chest pain and shortness of breath can be concerning. Based on these symptoms, I recommend checking for **Heart Disease** risk factors immediately."
    },
    "diabetes": {
        "keywords": ["thirsty", "drinking water", "urination", "toilet often", "hungry", "weight loss", "blurry vision", "tired", "slow healing", "wounds"],
        "msg": "Excessive thirst, frequent urination, and unexplained weight loss are classic indicators of **Diabetes**. It would be wise to screen for this."
    },
    "hepatitis": {
        "keywords": ["yellow skin", "yellow eyes", "jaundice", "stomach pain", "abdominal pain", "vomiting", "nausea", "dark urine", "pale stool"],
        "msg": "The yellowing of skin or eyes (jaundice) combined with abdominal pain strongly suggests issues with the liver, such as **Hepatitis**."
    },
    "breast_cancer": {
        "keywords": ["lump breast", "lump chest", "breast pain", "nipple discharge", "swelling underarm", "change in breast shape"],
        "msg": "Any new lumps, pain, or changes in breast tissue should be taken seriously. Please proceed with the **Breast Cancer** risk assessment."
    },
    "parkinsons": {
        "keywords": ["shaking hands", "tremor", "stiff muscles", "slow movement", "balance problems", "slurred speech", "hand writing small"],
        "msg": "Tremors, stiffness, and slow movement are characteristic signs of **Parkinson's Disease**. Let's assess your risk factors."
    },
    "malaria": {
        "keywords": ["high fever", "shivering", "chills", "sweating", "headache", "nausea", "vomiting", "muscle pain", "body ache"],
        "msg": "High fever accompanied by severe chills and shivering is typical of **Malaria**, especially if you've been in high-risk areas."
    },
    "flu": {
        "keywords": ["runny nose", "sneezing", "stuffy nose", "sore throat", "fever", "cough", "body aches", "feeling cold"],
        "msg": "These symptoms (fever, aches, congestion) closely match **Influenza (Flu)**. Rest is usually recommended, but let's check your vitals."
    },
    "pneumonia": {
        "keywords": ["cough with phlegm", "mucus", "green phlegm", "chest pain when breathing", "shortness of breath", "fever", "sweating", "chills"],
        "msg": "A productive cough with chest pain and fever could indicate **Pneumonia**, which requires medical attention."
    },
    "tuberculosis": {
        "keywords": ["coughing blood", "long cough", "cough 3 weeks", "night sweats", "weight loss", "chest pain", "weakness"],
        "msg": "A persistent cough (lasting weeks), especially with night sweats or blood, warrants a specific check for **Tuberculosis (TB)**."
    },
    "arthritis": {
        "keywords": ["joint pain", "stiff joints", "swollen joints", "knee pain", "finger pain", "hard to move", "red joints"],
        "msg": "Pain, stiffness, and swelling in the joints are key symptoms of **Arthritis**."
    },
    "asthma": {
        "keywords": ["wheezing", "whistling breath", "chest tight", "breathless", "coughing at night", "allergies", "dust allergy"],
        "msg": "Wheezing and chest tightness, particularly at night or with exertion, are signs of **Asthma**."
    },
    "migraine": {
        "keywords": ["severe headache", "throbbing head", "pain on one side", "light sensitivity", "sound sensitivity", "nausea", "visual aura"],
        "msg": "A severe, throbbing headache with sensitivity to light or sound suggests a **Migraine** rather than a typical tension headache."
    },
    "alzheimers": {
        "keywords": ["forgetting names", "memory loss", "confusion", "getting lost", "asking same questions", "personality change"],
        "msg": "Significant memory loss, confusion about time or place, and repetitive questioning can be early signs of **Alzheimer's**."
    },
    "anemia": {
        "keywords": ["pale skin", "feeling weak", "constant tiredness", "dizzy", "lightheaded", "cold hands", "cold feet", "short breath"],
        "msg": "Looking pale, feeling constantly tired, and having cold extremities often points to **Anemia** (low iron/hemoglobin)."
    },
    "depression": {
        "keywords": ["feeling sad", "hopeless", "crying", "lost interest", "sleeping too much", "insomnia", "suicidal", "guilt"],
        "msg": "Persistent sadness, loss of interest, and changes in sleep are symptoms of **Depression**. It's important to take this seriously."
    },
    "anxiety": {
        "keywords": ["worrying", "nervous", "panic attack", "racing heart", "sweating", "trembling", "impending doom"],
        "msg": "Excessive, uncontrollable worry and physical signs like a racing heart are characteristic of **Anxiety** disorders."
    },
    "uti": {
        "keywords": ["burning pee", "painful urination", "urinate often", "cloudy urine", "strong smelling urine", "pelvic pain"],
        "msg": "A burning sensation during urination and frequent urges to go usually indicate a **Urinary Tract Infection (UTI)**."
    },
    "gerd": {
        "keywords": ["heartburn", "acid reflux", "chest burning", "sour taste", "burping", "trouble swallowing", "lump in throat"],
        "msg": "Frequent heartburn and a sour taste/acid coming up are classic signs of **GERD** (Acid Reflux)."
    },
    "copd": {
        "keywords": ["smokers cough", "chronic cough", "lot of mucus", "wheezing", "tight chest", "blue lips"],
        "msg": "Chronic coughing and breathing difficulties, especially with a history of smoking, are linked to **COPD**."
    },
    "psoriasis": {
        "keywords": ["red patches skin", "silvery scales", "thick skin", "itching skin", "cracked skin", "joint pain"],
        "msg": "Thick red patches of skin covered with silvery scales are distinctive of **Psoriasis**."
    },
    "osteoporosis": {
        "keywords": ["bone fracture", "broken bone", "back pain", "getting shorter", "stooped posture"],
        "msg": "Frequent fractures from minor falls or loss of height may indicate **Osteoporosis** (brittle bones)."
    },
    "gout": {
        "keywords": ["pain in big toe", "swollen toe", "red toe", "hot joint", "sudden intense pain"],
        "msg": "Sudden, severe pain, redness, and swelling in the big toe is the most common presentation of **Gout**."
    },
    "eczema": {
        "keywords": ["itchy skin", "dry skin", "red rash", "scaly skin", "skin infection"],
        "msg": "Intensely itchy, red, and dry skin patches are typical of **Eczema**."
    },
     "sleep_apnea": {
        "keywords": ["loud snoring", "choking in sleep", "gasping sleep", "tired during day", "morning headache"],
        "msg": "Loud snoring followed by gasping or pauses in breathing suggests **Sleep Apnea**."
    },
      "kidney": {
        "keywords": ["swollen feet", "swollen ankles", "puffy eyes", "tired", "foamy urine", "dry itchy skin"],
        "msg": "Swelling in the ankles/feet, fatigue, and changes in urination can signal **Kidney Disease**."
    },
      "liver": {
        "keywords": ["yellow skin", "swollen abdomen", "easy bruising", "itchy skin", "dark urine"],
        "msg": "Abdominal swelling, yellow skin, and easy bruising are advanced warnings for **Liver Disease**."
    }
}

def analyze_symptoms(text):
    text = text.lower()
    
    # 1. Pre-checks for Greetings/General
    greetings = ["hi", "hello", "hey", "start", "help"]
    if any(w in text.split() for w in greetings):
        return None, "Hello! I'm here to help. Please tell me about your physical symptoms (e.g., 'I have a headache', 'My chest hurts')."

    scores = {}
    
    # 2. Score diseases
    for disease, data in SYMPTOM_DB.items():
        score = 0
        matched_keywords = []
        for kw in data['keywords']:
            # Check for multi-word phrases or single words
            if kw in text:
                score += 1
                matched_keywords.append(kw)
        
        # Bonus for multiple different keyword matches
        if score > 0:
            scores[disease] = score
            
    if not scores:
        return None, "I'm listening, but I didn't catch specific symptoms I recognize safely. Could you describe where it hurts or how you feel in more detail? (e.g., 'I feel dizzy', 'I have a fever')"
    
    # 3. Get best match
    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]
    
    # 4. Confidence Threshold (Optional logic can be added here)
    
    return best_match, SYMPTOM_DB[best_match]['msg']

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------

FEATURE_CONFIG = {
    # --- COMMON ---
    "Fever": {"label": "Body Temperature", "type": "basic", "help": "Body temperature in Fahrenheit."},
    "Cough": {"label": "Cough Severity", "type": "basic", "help": "Scale 0-10"},
    "Fatigue": {"label": "Fatigue Level", "type": "basic", "help": "Scale 0-10"},
    "Age": {"label": "Age", "type": "basic", "help": "Patient Age"},
    "BMI": {"label": "BMI", "type": "basic", "help": "Body Mass Index"},
    
    # --- HEART ---
    "age": {"label": "Age", "type": "basic", "help": "Age of the patient in years."},
    "sex": {"label": "Sex", "type": "basic", "help": "1 = Male, 0 = Female"},
    "cp": {"label": "Chest Pain Type", "type": "basic", "help": "0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"},
    "trestbps": {"label": "Resting Blood Pressure", "type": "basic", "help": "Resting blood pressure (mm Hg). Normal is < 120/80."},
    "chol": {"label": "Serum Cholesterol", "type": "basic", "help": "Serum cholesterol in mg/dl."},
    "fbs": {"label": "Fasting Blood Sugar > 120", "type": "basic", "help": "1 = True (>120 mg/dl), 0 = False"},
    "restecg": {"label": "Resting ECG", "type": "advanced", "help": "ECG results."},
    "thalach": {"label": "Max Heart Rate", "type": "advanced", "help": "Maximum heart rate achieved."},
    "exang": {"label": "Exercise Induced Angina", "type": "advanced", "help": "1 = Yes, 0 = No"},
    "oldpeak": {"label": "ST Depression", "type": "advanced", "help": "ST depression induced by exercise."},
    "slope": {"label": "Slope of ST", "type": "advanced", "help": "Peak exercise ST segment slope."},
    "ca": {"label": "Major Vessels (0-3)", "type": "advanced", "help": "Number of major vessels colored by flourosopy."},
    "thal": {"label": "Thal Error", "type": "advanced", "help": "Thalassemia status."},

    # --- DIABETES ---
    "Pregnancies": {"label": "Pregnancies", "type": "basic", "help": "Number of times pregnant."},
    "Glucose": {"label": "Glucose Level", "type": "basic", "help": "Plasma glucose concentration."},
    "BloodPressure": {"label": "Blood Pressure", "type": "basic", "help": "Diastolic blood pressure (mm Hg)."},
    "SkinThickness": {"label": "Skin Thickness", "type": "advanced", "help": "Triceps skin fold thickness (mm)."},
    "Insulin": {"label": "Insulin Level", "type": "advanced", "help": "2-Hour serum insulin."},
    "DiabetesPedigreeFunction": {"label": "Pedigree Function", "type": "advanced", "help": "Diabetes pedigree function."},


    # --- HEPATITIS ---
    "Age": {"label": "Age", "type": "basic", "help": "Patient Age"},
    "Sex": {"label": "Sex", "type": "basic", "help": "1 = Male, 2 = Female"},
    "Fatigue": {"label": "Fatigue", "type": "basic", "help": "Flu-like symptom."},
    "Malaise": {"label": "Malaise", "type": "basic", "help": "General feeling of discomfort."},
    "Anorexia": {"label": "Anorexia", "type": "basic", "help": "Loss of appetite."},
    
    # Hepatitis - Advanced
    "Steroid": {"label": "Steroids", "type": "advanced"},
    "Antivirals": {"label": "Antivirals", "type": "advanced"},
    "Liver Big": {"label": "Liver Big", "type": "advanced"},
    "Liver Firm": {"label": "Liver Firm", "type": "advanced"},
    "Spleen Palpable": {"label": "Spleen Palpable", "type": "advanced"},
    "Spiders": {"label": "Spiders", "type": "advanced"},
    "Ascites": {"label": "Ascites", "type": "advanced"},
    "Varices": {"label": "Varices", "type": "advanced"},
    "Bilirubin": {"label": "Bilirubin", "type": "advanced"},
    "Alk Phosphate": {"label": "Alk Phosphate", "type": "advanced"},
    "Sgot": {"label": "SGOT", "type": "advanced"},
    "Albumin": {"label": "Albumin", "type": "advanced"},
    "Protime": {"label": "Protime", "type": "advanced"},
    "Histology": {"label": "Histology", "type": "advanced"},
}

def get_feature_info(name):
    """Retrieve label, type and help text."""
    clean_name = name.strip()
    if clean_name in FEATURE_CONFIG:
        return FEATURE_CONFIG[clean_name]
    
    return {
        "label": clean_name.replace("_", " ").title(), 
        "type": "basic",
        "help": f"Enter value for {clean_name}."
    }

# -----------------------------------------------------------------------------
# APP LOAD
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

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("MedDiagnostic AI")
    st.markdown("---")
    
    # Navigation
    app_mode = st.radio("Navigation", ["💬 AI Symptom Chat", "📝 Clinical Assessment"])
    
    st.markdown("---")
    
    if app_mode == "📝 Clinical Assessment":
        st.subheader("Mode Selection")
        user_mode = st.radio(
            "User Type",
            ["General User", "Medical Professional"],
            index=0
        )
        is_basic_mode = (user_mode == "General User")
        
        st.markdown("---")
        
        display_map = {k: k.replace('_', ' ').title() for k in models.keys()}
        inv_display_map = {v: k for k, v in display_map.items()}
        
        # Initialize session state for selection if not present
        if 'selected_disease_name' not in st.session_state:
            st.session_state.selected_disease_name = list(display_map.values())[0]

        selected_display_name = st.selectbox(
            "Select Condition", 
            list(display_map.values()),
            index=list(display_map.values()).index(st.session_state.selected_disease_name)
        )
        selected_disease = inv_display_map[selected_display_name]
        
        # Update session state
        st.session_state.selected_disease_name = selected_display_name
        
        acc = accuracies.get(selected_disease, 0.0)
        st.metric("Model Reliability", f"{acc:.1%}")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------

if app_mode == "💬 AI Symptom Chat":
    st.title("🤖 AI Health Assistant")
    st.markdown("Describe your symptoms below, and I will recommend the appropriate clinical screening.")
    
    # Chat History Container
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your medical assistant. How are you feeling today? Tell me about any symptoms you are experiencing."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ex: I have chest pain and feel dizzy..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Analysis
        disease_match, response_text = analyze_symptoms(prompt)
        
        # Simulate thinking
        with st.spinner("Analyzing symptoms..."):
            time.sleep(1)
            
        # Assistant Response
        with st.chat_message("assistant"):
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            if disease_match:
                # Button to redirect
                display_name = disease_match.replace('_', ' ').title()
                if st.button(f"Start {display_name} Assessment →"):
                    st.session_state.selected_disease_name = display_name
                    # Force a rerun to switch tabs? 
                    # Streamlit doesn't strictly allow switching tabs programmatically easily without session state tricks
                    # We will ask user to switch manually or just highlight it
                    st.info(f"Please switch to the 'Clinical Assessment' tab in the sidebar and select **{display_name}** to begin your diagnosis.")

elif app_mode == "📝 Clinical Assessment":
    st.title(f"{st.session_state.selected_disease_name} Assessment")

    if is_basic_mode:
        st.info("ℹ️ **Simplified Mode**: Estimating complex lab values using population averages.")
    
    with st.form("diagnostic_form"):
        st.subheader("Patient Vitals & History")
        
        features = feature_names.get(selected_disease, [])
        stats = feature_stats.get(selected_disease, {})
        
        visible, hidden = [], []
        for feat in features:
            info = get_feature_info(feat)
            if is_basic_mode and info.get('type') == 'advanced':
                hidden.append(feat)
            else:
                visible.append(feat)
                
        # Visible Inputs
        cols = st.columns(3)
        input_values = {}
        
        for i, feat in enumerate(visible):
            with cols[i % 3]:
                info = get_feature_info(feat)
                feat_stat = stats.get(feat, {})
                min_v, max_v, mean_v = feat_stat.get('min', 0.0), feat_stat.get('max', 10.0), feat_stat.get('mean', 5.0)
                rng = max_v - min_v if (max_v - min_v) > 0 else 10.0
                
                val = st.number_input(
                    label=info['label'],
                    min_value=float(min_v - rng*0.2),
                    max_value=float(max_v + rng*0.2),
                    value=float(mean_v),
                    format="%.2f",
                    help=info['help'],
                    key=feat
                )
                input_values[feat] = val
                
        # Hidden inputs
        for feat in hidden:
            input_values[feat] = stats.get(feat, {}).get('mean', 0.0)

        st.markdown("---")
        submit = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)

    if submit:
        try:
            # Build vector
            input_list = [input_values[f] for f in features]
            input_vector = pd.DataFrame([input_list], columns=features)
            
            # Predict
            model = models[selected_disease]
            if selected_disease in scalers:
                input_vector = scalers[selected_disease].transform(input_vector)
                
            pred = model.predict(input_vector)[0]
            prob = model.predict_proba(input_vector)[0, 1] if hasattr(model, "predict_proba") else 0.0
            
            # Results
            st.markdown("### Results")
            res1, res2 = st.columns([2, 1])
            with res1:
                if pred == 1:
                    st.error("⚠️ HIGH RISK")
                    st.markdown("**Action Required:** Please consult a specialist immediately.")
                else:
                    st.success("✅ LOW RISK")
                    st.markdown("No significant risk factors detected.")
            with res2:
                st.metric("Probability", f"{prob:.1%}")
                st.progress(prob)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
