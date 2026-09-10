"""
Data Loader & Preprocessing Module for Post-hoc Explanation Project.
Handles dataset generation/loading, preprocessing, train-test splitting, and scaling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = Male, 0 = Female)',
    'cp': 'Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
    'restecg': 'Resting ECG Results (0: Normal, 1: ST-T Abnormality, 2: LVH)',
    'thalach': 'Maximum Heart Rate Achieved (bpm)',
    'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
    'oldpeak': 'ST Depression Induced by Exercise',
    'slope': 'Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)',
    'ca': 'Major Vessels Colored by Fluoroscopy (0-4)',
    'thal': 'Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversable Defect)'
}

TARGET_NAME = 'target'


def generate_synthetic_heart_data(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic synthetic heart disease dataset with non-linear feature interactions."""
    np.random.seed(random_state)
    
    age = np.random.randint(30, 78, size=n_samples)
    sex = np.random.binomial(1, 0.65, size=n_samples)
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.18, 0.27, 0.10])
    trestbps = np.random.randint(94, 200, size=n_samples)
    chol = np.random.randint(130, 420, size=n_samples)
    fbs = np.random.binomial(1, 0.15, size=n_samples)
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.45, 0.05])
    thalach = np.clip(208 - 0.7 * age + np.random.normal(0, 15, size=n_samples), 70, 202).astype(int)
    exang = np.random.binomial(1, 0.32, size=n_samples)
    oldpeak = np.round(np.clip(np.random.exponential(1.1, size=n_samples), 0, 6.2), 1)
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.46, 0.46, 0.08])
    ca = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.58, 0.20, 0.12, 0.07, 0.03])
    thal = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.06, 0.39])

    # Realistic non-linear logit computation for risk target
    logit = (
        0.04 * (age - 54) +
        0.5 * sex +
        0.8 * cp +
        0.015 * (trestbps - 130) +
        0.005 * (chol - 240) +
        0.4 * fbs +
        0.3 * restecg -
        0.035 * (thalach - 150) +
        0.9 * exang +
        0.75 * oldpeak +
        0.4 * slope +
        0.65 * ca +
        0.85 * thal -
        1.2 # Intercept
    )
    
    # Non-linear interaction terms
    logit += 0.3 * (oldpeak * exang) - 0.02 * (thalach * (cp > 0))

    probs = 1 / (1 + np.exp(-logit))
    target = (probs >= 0.5).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    
    return df


def load_dataset(csv_path: str = 'data/heart_disease.csv'):
    """Loads dataset from CSV or generates synthetic dataset if missing."""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = generate_synthetic_heart_data()
        df.to_csv(csv_path, index=False)
        print(f"[DataLoader] Generated synthetic dataset saved to {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"[DataLoader] Loaded dataset from {csv_path}")
        
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return df, X, y


def get_prepared_data(test_size: float = 0.2, random_state: int = 42):
    """Returns train/test splits, scaler, and original DataFrames."""
    df, X, y = load_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_NAMES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_NAMES, index=X_test.index)
    
    return {
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': FEATURE_NAMES,
        'feature_descriptions': FEATURE_DESCRIPTIONS
    }


if __name__ == '__main__':
    data_dict = get_prepared_data()
    print("Dataset Shape:", data_dict['df'].shape)
    print("Class Balance:", data_dict['y'].value_counts(normalize=True).to_dict())
