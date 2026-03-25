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

    # 4. Sauvegarde des métriques et du modèle
    os.makedirs('models', exist_ok=True)
    results_df.to_csv('models/model_metrics.csv', index=False)
    
    best_model = trained_models[best_model_name]
    save_model(best_model, 'models/best_model.pkl')

if __name__ == '__main__':
    main()
