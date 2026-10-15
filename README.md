# ✍️ MNIST Digit Recognizer & Interactive Canvas

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Application web interactive de reconnaissance de chiffres manuscrits combinant un canevas de dessin tactile, un pipeline de prétraitement d'image haute fidélité (aligné sur les spécifications de Yann LeCun) et un réseau de neurones artificiels (MLP).

---

## 🌟 Points Forts du Projet

- **🎨 Dessin en direct** : Interface fluide grâce à `streamlit-drawable-canvas` avec ajustement de l'épaisseur du trait.
- **🔬 Pipeline de normalisation rigoureux** :
  - Détection automatique de la boîte englobante (*bounding box*).
  - Redimensionnement homothétique dans une zone de **20×20 px** (conservation du ratio).
  - Intégration sur canevas **28×28 px**.
  - **Recentrage par centre de masse** (*Center of Mass*) via `scipy.ndimage`, reproduisant fidèlement la méthodologie du dataset MNIST original.
- **⚡ Modèle Léger & Performant** : Réseau de neurones multi-couches (MLP) atteignant **~97.5% de précision**, persisté dans `mnist_model.joblib` et chargé instantanément via `@st.cache_resource`.
- **📊 Interprétabilité & Métriques** : Affichage du chiffre prédit, du niveau de confiance, du graphique de distribution des probabilités sur les 10 classes et du Top 3 des prédictions.

---

## 📁 Structure du Projet

```text
mnist-streamlit/
│── app.py                  # Application Streamlit principale et logique ML
│── mnist_model.joblib      # Modèle entraîné (MLPClassifier sérialisé)
│── requirements.txt        # Dépendances du projet
│── .gitignore              # Fichiers à exclure de git
│── LICENSE                 # Licence MIT
└── README.md               # Documentation du projet
```

---

## 🚀 Démarrage Rapide (En local)

### 1. Cloner le dépôt
```bash
git clone https://github.com/VOTRE_PSEUDO/mnist-streamlit.git
cd mnist-streamlit
```

### 2. Créer et activer un environnement virtuel (optionnel mais recommandé)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'application
```bash
streamlit run app.py
```
L'interface s'ouvrira directement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## ☁️ Déploiement Gratuit sur Streamlit Community Cloud

Ce projet est prêt à être déployé en 1 clic gratuitement :
1. Créez un compte sur [share.streamlit.io](https://share.streamlit.io/) en vous connectant avec votre compte GitHub.
2. Cliquez sur **"New app"**.
3. Sélectionnez ce dépôt (`VOTRE_PSEUDO/mnist-streamlit`), branche `main` et fichier `app.py`.
4. Cliquez sur **"Deploy"**. Vous obtiendrez un lien public interactif à ajouter directement sur votre CV ou profil LinkedIn !

---

## 🛠️ Technologies Utilisées

- **Frontend & Déploiement** : [Streamlit](https://streamlit.io/)
- **Composant de Dessin** : [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
- **Machine Learning** : [Scikit-learn](https://scikit-learn.org/) (MLPClassifier)
- **Traitement d'Image & Calcul** : [NumPy](https://numpy.org/), [Pillow (PIL)](https://python-pillow.org/), [SciPy](https://scipy.org/)
- **Visualisation & Données** : [Pandas](https://pandas.pydata.org/)

---

## 📜 Licence

Ce projet est sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.
