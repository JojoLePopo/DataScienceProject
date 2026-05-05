from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

def define_models():
    # 4 modèles avec des hyperparamètres optimisés pour mieux gérer le déséquilibre et la complexité
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', C=0.1, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        'MLP_DeepLearning': MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.001, max_iter=1000, early_stopping=True, random_state=42)
    }
    return models

def evaluate_model(y_true, y_pred, y_prob=None):
    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try:
            results['ROC-AUC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
    return results

def train_and_compare_models(X_train, X_test, y_train, y_test, preprocessor):
    models = define_models()
    results_df = []
    trained_models = {}

    for name, model in models.items():
        print(f"Formation du modèle {name}...")
        
        # Pipeline combinant preprocessing et modèle
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Probabilités (certains classifieurs n'en ont pas ou sous condition)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
            
        res = evaluate_model(y_test, y_pred, y_prob)
        res['Model'] = name
        results_df.append(res)
        trained_models[name] = clf
        
    return pd.DataFrame(results_df), trained_models

def save_model(model, filepath='models/best_model.pkl'):
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans {filepath}")
