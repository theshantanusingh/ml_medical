import pandas as pd
import numpy as np
import os

# Disease feature definitions (Simplified for Synthetic Generation)
# Format: Name -> [Features...]
DISEASE_SCHEMA = {
    "Malaria": ["Fever", "Vomiting", "Headache", "Muscle Pain", "Sweating"],
    "Tuberculosis": ["Cough_Duration_Weeks", "Fever", "Weight_Loss", "Chest_Pain", "Night_Sweats"],
    "Flu": ["Fever", "Cough", "Sore_Throat", "Runny_Nose", "Fatigue"],
    "Arthritis": ["Joint_Pain", "Stiffness", "Swelling", "Age", "Previous_Injury"],
    "Asthma": ["Shortness_of_Breath", "Wheezing", "Coughing", "Chest_Tightness", "Allergy_History"],
    "Hypertension": ["Systolic_BP", "Diastolic_BP", "Age", "BMI", "Heart_Rate"],
    "Anemia": ["Hemoglobin", "Fatigue", "Pale_Skin", "Dizziness", "Iron_Level"],
    "Migraine": ["Headache_Intensity", "Nausea", "Light_Sensitivity", "Sound_Sensitivity", "Duration_Hours"],
    "Alzheimers": ["Memory_Loss_Score", "Confusion", "Age", "Disorientation", "Personality_Change"],
    "Osteoporosis": ["Bone_Density_Score", "Age", "Previous_Fractures", "Calcium_Intake", "Vitamin_D_Level"],
    "Psoriasis": ["Red_Patches", "Scaling", "Itching", "Joint_Pain", "Nail_Changes"],
    "Gout": ["Uric_Acid", "Joint_Pain_Toe", "Swelling", "Redness", "Alcohol_Consumption"],
    "GERD": ["Heartburn", "Regurgitation", "Chest_Pain", "Difficulty_Swallowing", "Cough"],
    "COPD": ["Smoking_History_Years", "Shortness_of_Breath", "Cough", "Sputum_Production", "Age"],
    "UTI": ["Urination_Frequency", "Burning_Sensation", "Pelvic_Pain", "Urine_Cloudy", "Fever"],
    "Depression": ["Sadness_Score", "Interest_Loss", "Sleep_Disturbance", "Energy_Level", "Concentration"],
    "Anxiety": ["Worry_Score", "Restlessness", "Fatigue", "Irritability", "Muscle_Tension"],
    "Sleep_Apnea": ["Snoring_Loudness", "Tiredness", "Observed_Apnea", "BP", "BMI"],
    "Eczema": ["Itching", "Redness", "Dry_Skin", "Family_History", "Asthma_History"],
    "Pneumonia": ["Fever", "Cough_Phlegm", "Chest_Pain_Breathing", "Fatigue", "Nausea"]
}

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dataset(name, features, n_samples=500):
    data = {}
    
    # Generate realistic-looking synthetic data
    for feat in features:
        if "Age" in feat:
            data[feat] = np.random.normal(50, 20, n_samples).clip(10, 90)
        elif "BP" in feat or "Pressure" in feat:
            data[feat] = np.random.normal(120, 20, n_samples)
        elif "BMI" in feat:
            data[feat] = np.random.normal(25, 5, n_samples)
        elif "Score" in feat or "Level" in feat:
            data[feat] = np.random.uniform(0, 10, n_samples)
        elif "History" in feat or "Previous" in feat:
            data[feat] = np.random.randint(0, 2, n_samples) # Binary
        elif "Intensity" in feat or "Duration" in feat:
            data[feat] = np.random.exponential(5, n_samples)
        else:
            # Assume symptom severity or presence (0-10 or binary)
            # using continuous for better model training compatibility with existing regression/classification logic
            data[feat] = np.random.uniform(0, 10, n_samples)
            
    df = pd.DataFrame(data)
    
    # Generate Target (Synthetic logic: average of features + noise > threshold)
    # This ensures the model can actually "learn" something
    numeric_df = df.select_dtypes(include=[np.number])
    score = numeric_df.mean(axis=1) + np.random.normal(0, 1, n_samples)
    threshold = score.median()
    df['target'] = (score > threshold).astype(int)
    
    # Save
    filename = f"{name.lower().replace(' ', '_')}.csv"
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Generated {filename} with 500 samples.")

def main():
    print("Generating synthetic medical datasets...")
    for disease, features in DISEASE_SCHEMA.items():
        generate_dataset(disease, features)
    print("Done!")

if __name__ == "__main__":
    main()
