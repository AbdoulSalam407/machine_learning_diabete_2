# Prédiction du diabète (PIMA Indians) — Machine Learning

Projet d’**apprentissage statistique** (classification binaire) visant à prédire si une patiente est atteinte d’un **diabète de type 2** à partir de mesures cliniques.

- **Cible**: `Outcome` (`0` = non diabétique, `1` = diabétique)
- **Dataset**: PIMA Indians Diabetes Database (Kaggle / UCI)

## Contenu du dépôt

- **Notebook principal**: `projet_ML_prediction_diabete_2_ASD_AF_Python.ipynb`
- **Rapport**: `projet_machine_learning_diabete.pdf` / `projet_machine_learning_diabete.docx`

## Données

Le notebook charge un fichier local:

- `diabetes.csv`

Tu peux le récupérer via Kaggle:

- https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Ensuite, place `diabetes.csv` **à la racine du projet** (au même niveau que le notebook).

### Variables (features)

Le dataset contient 768 observations et 9 colonnes:

- `Pregnancies`: nombre de grossesses
- `Glucose`: glucose plasmatique à jeun
- `BloodPressure`: pression artérielle diastolique
- `SkinThickness`: épaisseur du pli cutané tricipital
- `Insulin`: insuline sérique (2h)
- `BMI`: indice de masse corporelle
- `DiabetesPedigreeFunction`: indice de risque génétique
- `Age`: âge
- `Outcome`: variable cible

Remarque importante: certaines colonnes contiennent des **valeurs à 0** (ex: `Insulin`, `SkinThickness`, `BloodPressure`, `BMI`, `Glucose`) qui sont **biologiquement impossibles** et sont traitées comme des valeurs manquantes / incohérentes dans l’analyse.

## Approche / pipeline

Le notebook couvre notamment:

- Analyse exploratoire (EDA) et visualisations
- Identification de valeurs aberrantes (zéros invalides)
- Prétraitement (ex: standardisation selon les modèles)
- Gestion du déséquilibre de classes (ex: **SMOTE**)
- Entraînement et comparaison de modèles de classification
- Évaluation via métriques de classification (accuracy, matrice de confusion, ROC-AUC, courbe ROC, etc.)

### Modèles utilisés (selon le notebook)

- Régression logistique (`LogisticRegression`)
- SVM (`SVC`)
- Arbre de décision (`DecisionTreeClassifier`)
- Forêt aléatoire (`RandomForestClassifier`)
- Gradient boosting (`xgboost`)

## Installation

Le dépôt ne contient pas encore de `requirements.txt`. Voici une base de dépendances cohérente avec le notebook.

### Option A — Installation rapide (pip)

Dans un environnement virtuel Python (recommandé):

```bash
pip install -U pip
pip install pandas numpy matplotlib seaborn missingno scipy scikit-learn imbalanced-learn xgboost jupyter
```

### Option B — Générer un requirements.txt

Après installation, tu peux figer tes versions:

```bash
pip freeze > requirements.txt
```

## Exécution

1. Télécharger `diabetes.csv` et le placer à la racine du projet.
2. Lancer Jupyter:

```bash
jupyter notebook
```

3. Ouvrir et exécuter:

- `projet_ML_prediction_diabete_2_ASD_AF_Python.ipynb`

## Résultats

Les résultats (métriques, matrices de confusion, ROC-AUC, courbes ROC, sélection de modèle, etc.) sont générés directement dans le notebook.

Pour une synthèse complète, voir:

- `projet_machine_learning_diabete.pdf`

## Auteurs

- Awa FAYE
- Abdoul Salam DIALLO
