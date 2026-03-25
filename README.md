# Projet Data Science - Maintenance Prédictive Industrielle

Ce projet a été réalisé dans le cadre du module *Système Intelligent Multi-Modèles pour la Maintenance Prédictive Industrielle*.
L'objectif est d'utiliser des algorithmes de Machine Learning (ML) et Deep Learning (DL) pour la maintenance prédictive, en anticipant les pannes d'équipement industriel dans les 24h.

**Approche suivie (Simple et Minimaliste) :**
- **Sujet choisi :** Classification Binaire (Prédire la panne dans les 24h).
- **Modèles utilisés :** Régression Logistique, RandomForest, Gradient Boosting, et MLP (Perceptron Multicouche en tant que modèle Deep Learning).
- **Interface :** Dashboard interactif sous Streamlit.

## 🛠️ Structure du Projet

```text
_PROJET DATA SCIENCE/
│
├── data/                    # Dossier contenant le(s) dataset(s)
│   └── industrial_machine_maintenance.csv  (À télécharger depuis Kaggle)
│
├── src/                     # Code source métier
│   ├── data_preprocessing.py # Script de nettoyage et preprocessing (Pipelines sklearn)
│   └── modeling.py           # Script d'entraînement et évaluation des modèles
│
├── models/                  # Contient le meilleur modèle après entraînement (.pkl)
│
├── api.py                   # API REST FastAPI pour exposer le modèle
├── app.py                   # Dashboard Streamlit interactif
├── main.py                  # Point d'entrée pour l'entraînement (CLI)
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation (Ce fichier)

```

## 🚀 Installation & Exécution

### 1 Prérequis
- Python 3.9+ 
- Installer les dépendances :
  ```bash
  pip install -r requirements.txt
  ```


### 2 Entraîner les modèles (Pipeline complet)
Pour lancer le processus de Data Science (chargement, pré-traitement, entraînement de 4 modèles dont 1 DL, évaluation, comparaison, et sauvegarde du meilleur modèle) :
```bash
python main.py
```
Le résultat s'affichera dans la console (Accuracy, F1-Score, etc.) et le meilleur modèle sera sauvegardé dans `models/best_model.pkl`.

### 4️⃣ Lancer l'API REST (Industrialisation)
Pour exposer votre modèle de manière professionnelle, lancez l'API FastAPI :
```bash
uvicorn api:app --reload
```
- L'API sera accessible sur `http://127.0.0.1:8000`.
- Vous pouvez tester l'interface interactive et voir la documentation en allant sur `http://127.0.0.1:8000/docs`.

### 5️⃣ Lancer le Dashboard interactif
Pour visualiser le résultat métier (orienté pour un responsable de maintenance) via une interface simple :
```bash
streamlit run app.py
```
Une fenêtre navigateur s'ouvrira (généralement `http://localhost:8501`), vous permettant de simuler l'état d'un équipement (vibration, température, etc.) et d'obtenir en direct une prédiction de panne.

## 📈 Méthodes et choix d'implémentation
1. **Prévention du Data Leakage :** Utilisation systématique de `Pipeline` et `ColumnTransformer` via Scikit-Learn.
2. **Imputation et Encodage :** Standardisation (StandardScaler) pour les variables numériques, et OneHotEncoding pour les variables catégorielles (Mode Opératoire).
3. **Métrique privilégiée :** Le F1-Score est privilégié lors de la sélection du meilleur modèle, du fait du déséquilibre très courant inhérent aux problémes de maintenance industrielle (les pannes sont heureusement rares).
