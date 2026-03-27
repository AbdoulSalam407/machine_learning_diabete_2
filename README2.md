# 1. Créer un dossier pour votre projet

mkdir projet_diabete
cd projet_diabete

# 2. Copier les fichiers dans ce dossier

# - model_diabete.pkl (téléchargé depuis Colab)

# - model_metadata.pkl (téléchargé depuis Colab)

# - app.py (le fichier ci-dessus)

# 3. Installer streamlit (si pas déjà fait)

pip install streamlit pandas numpy plotly joblib

# 4. Lancer l'application

streamlit run app.py
