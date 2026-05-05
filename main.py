# main.py
import os
from src.data_preprocessing import load_and_preprocess_data
from src.modeling import train_and_compare_models, save_model

def main():
    print("Démarrage du pipeline de Maintenance Prédictive Industrielle (Classification Binaire)...")
    
    # 1. Chargement et Préparation des Données
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('data/predictive_maintenance_v3.csv')
    print("Données chargées et prétraitées.")

    # 2. Entraînement et Comparaison de 4 Modèles (dont 1 DL : MLP)
    results_df, trained_models = train_and_compare_models(X_train, X_test, y_train, y_test, preprocessor)
    print("\nRésultats de l'évaluation comparative :")
    print(results_df.to_string(index=False))

    # 3. Choix du meilleur modèle basé sur le F1-Score (compromis Precision/Recall important en industrie)
    best_model_name = results_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
    print(f"\nMeilleur modèle sélectionné : {best_model_name}")
    
    # Ajout d'une colonne Global pour identifier le modèle choisi
    results_df['Global'] = results_df['Model'].apply(lambda x: '🏆 Sélectionné' if x == best_model_name else '')

    best_model = trained_models[best_model_name]

    # 4. Calculs post-entraînement : Matrice de Confusion et Feature Importance
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd

    # 4a. Matrice de confusion
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs('models', exist_ok=True)
    np.save('models/confusion_matrix.npy', cm)

    # 4b. Importance des variables ou Coefficients
    classifier = best_model.named_steps['classifier']
    importances = None
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = classifier.coef_[0]

    if importances is not None:
        try:
            feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            feature_names = [f.split('__')[-1] for f in feature_names] # Nettoyage des préfixes (ex: num__vibration -> vibration)
        except AttributeError:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficients': importances
        })
        coef_df.to_csv('models/model_coefficients.csv', index=False)

    # 5. Sauvegarde des métriques et du modèle
    results_df.to_csv('models/model_metrics.csv', index=False)
    save_model(best_model, 'models/best_model.pkl')

if __name__ == '__main__':
    main()
