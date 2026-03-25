import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(filepath='data/predictive_maintenance_v3.csv'):
    # Charger les données
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        import numpy as np
        print("Dataset introuvable. Création de données fictives...")
        np.random.seed(42)
        df = pd.DataFrame({
            'vibration_rms': np.random.uniform(0.1, 5.0, 1000),
            'temperature_motor': np.random.uniform(50, 120, 1000),
            'current_phase_avg': np.random.uniform(10, 50, 1000),
            'rpm': np.random.uniform(1000, 3000, 1000),
            'pressure_level': np.random.uniform(10, 50, 1000),
            'operating_mode': np.random.choice(['normal', 'idle', 'peak'], 1000),
            'hours_since_maintenance': np.random.uniform(0, 8000, 1000),
            'failure_within_24h': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
        })
    
    # 1. Feature Selection : on garde uniquement les variables les plus pertinentes pour le modèle
    # On exclut les identifiants, le timestamp, et les autres cibles (fuite de données)
    selected_features = [
        'vibration_rms', 
        'temperature_motor', 
        'current_phase_avg', 
        'pressure_level', 
        'rpm', 
        'hours_since_maintenance', 
        'operating_mode',
        'failure_within_24h'  # Cible à conserver pour l'instant
    ]
    
    # On filtre les colonnes qui existent bien dans le dataset
    df = df[[col for col in selected_features if col in df.columns]]
    
    # Séparation Features / Target
    X = df.drop(columns=['failure_within_24h'])
    y = df['failure_within_24h']
    
    # Identifier les types de variables
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Pipeline de transformation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Pour rester simple et minimaliste, on utilise un OneHotEncoder de pandas ou de sklearn
        # Ici on peut le faire plus simplement avec pd.get_dummies dans un vrai projet
    ])
    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor
