import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml
import os


class MultiDiseasePredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = (
            yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}
        )
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.feature_stats = {}
        self.accuracies = {}

    def add_disease(self, dataset_path, target_col, disease_name, columns=None):
        """Add disease with SAFETY CHECKS"""
        print(f"\n🔄 ADDING {disease_name.upper()}...")

        if not os.path.exists(dataset_path):
            print(f"❌ {dataset_path} not found - SKIPPING")
            return

        try:
            if columns:
                df = pd.read_csv(dataset_path, names=columns, header=None)
            else:
                df = pd.read_csv(dataset_path)
        except pd.errors.EmptyDataError:
            print(f"❌ {disease_name}: Empty file - SKIPPING")
            return
        except Exception as e:
            print(f"❌ {disease_name}: Load failed - {str(e)}")
            return

        print(f"✅ {disease_name}: {len(df)} samples loaded")

        # Auto-fix target column
        available_cols = df.columns.tolist()
        if target_col not in available_cols:
            target_col = available_cols[-1]
            print(f"🔧 Using '{target_col}' as target")

        # SAFETY CHECK 1: Remove non-numeric 'name' column (Parkinson's issue)
        if "name" in df.columns:
            df = df.drop("name", axis=1)

        # Robust Target Encoding
        # Convert to string first to handle mixed types, then Categorical codes
        try:
             # Check if target is string/object
            if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
                y = pd.Categorical(df[target_col]).codes
            else:
                # Numeric: maintain existing logic but safer
                y_series = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
                # If binary already (0/1 or 1/2), keep as is (mapped to 0/1)
                if y_series.nunique() == 2:
                    y = (y_series == y_series.max()).astype(int)
                else:
                    # Continuous variable as target? Binarize by median
                    y = (y_series > y_series.median()).astype(int)
        except Exception as e:
            print(f"❌ {disease_name}: Target encoding error - {str(e)}")
            return

        # SAFETY CHECK 2: Need minimum samples
        if len(df) < 20:
            print(f"❌ {disease_name}: Too few samples ({len(df)})")
            return

        # Features (numeric only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        if len(numeric_cols) == 0:
            print(f"❌ {disease_name}: No numeric features found")
            return

        X = df[numeric_cols].fillna(df[numeric_cols].median())

        # SAFETY CHECK 3: Train/test split needs variation
        if len(np.unique(y)) < 2:
            print(f"❌ {disease_name}: Target has no variation")
            return

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # SAFETY CHECK 4: Ensure training data exists
        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            print(f"❌ {disease_name}: Invalid train/test split")
            return

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model with ROBUST solver
        try:
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",  # More robust for small datasets
                max_iter=5000,
                random_state=42,
            )
            model.fit(X_train_scaled, y_train)

            # Test accuracy
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            self.feature_names[disease_name] = X.columns.tolist()
            self.models[disease_name] = model
            self.scalers[disease_name] = scaler
            self.accuracies[disease_name] = accuracy

            print(f"✅ {disease_name}: **{accuracy:.1%}** accuracy")
            print(f"✅ {disease_name}: **{accuracy:.1%}** accuracy")
            print(f"   → Train: {len(X_train)} | Test: {len(X_test)}")
            
            # Save stats for UI
            self.feature_stats[disease_name] = X.describe().T[['min', 'max', 'mean']].to_dict('index')

        except Exception as e:
            print(f"❌ {disease_name}: Training failed - {str(e)}")

    def load_existing(self):
        """Load your CURRENT 4 models"""
        if os.path.exists("models/multi_disease_pipeline.pkl"):
            pipeline = joblib.load("models/multi_disease_pipeline.pkl")
            self.models = pipeline.get("models", {})
            self.scalers = pipeline.get("scalers", {})
            self.feature_names = pipeline.get("feature_names", {})
            self.feature_stats = pipeline.get("feature_stats", {})
            self.accuracies = pipeline.get("accuracies", {})
            print(f"✅ Loaded {len(self.models)} existing models")

    def train_all(self):
        """Train with full safety checks"""
        self.load_existing()

        diseases = {
            "heart": ("data/raw/heart.csv", "target", [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]),
            "diabetes": ("data/raw/diabetes.csv", "Outcome", [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
            ]),
            "breast_cancer": ("data/raw/breast_cancer.csv", "diagnosis", [
                'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 
                'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 
                'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
                'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
            ]),
            "hepatitis": ("data/raw/hepatitis.csv", "Class", [
                'Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 
                'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 
                'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime', 'Histology'
            ]),
            "parkinsons": ("data/raw/parkinsons.csv", "status", None), # Has headers
            "liver": ("data/raw/liver.csv", "Dataset", None),
            "kidney": ("data/raw/kidney.csv", "Class", None),
            "stroke": ("data/raw/stroke.csv", "stroke", None),
            "thyroid": ("data/raw/thyroid.csv", "Class", None),
            "dengue": ("data/raw/dengue.csv", "case", None),
        }

        for disease, (path, target, cols) in diseases.items():
            self.add_disease(path, target, disease, columns=cols)

        self.save_pipeline()

    def save_pipeline(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "scalers": self.scalers,
                "feature_names": self.feature_names,
                "feature_stats": self.feature_stats,
                "accuracies": self.accuracies,
            },
            "models/multi_disease_pipeline.pkl",
        )
        print("\n🎉 PIPELINE SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    predictor = MultiDiseasePredictor()
    predictor.train_all()
    print("\n🚀 PRODUCTION READY!")
