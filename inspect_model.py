import joblib
import pandas as pd

try:
    pipeline = joblib.load('models/multi_disease_pipeline.pkl')
    feature_names = pipeline['feature_names']
    for disease, features in feature_names.items():
        print(f"--- {disease} ---")
        print(features[:5]) # just show first 5
except Exception as e:
    print(e)
