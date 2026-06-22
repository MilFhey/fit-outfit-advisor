# Fit & Outfit Advisor

Prototype universitaire MIAGE M2 - Réseaux de neurones / TensorFlow.

L'objectif est de créer un assistant intelligent d'achat vestimentaire capable de :

1. reconnaître un vêtement à partir d'une image produit ;
2. prédire si une taille risque d'être adaptée à l'utilisateur ;
3. proposer ou évaluer des associations de vêtements ;
4. générer un conseil final compréhensible.

## Stratégie MVP

Le projet est construit comme une cabine d'essayage intelligente :

- **Streamlit** sert de cabine visible par l'utilisateur ;
- les **services Python** jouent le rôle des assistants spécialisés ;
- les **modèles TensorFlow/Keras** seront branchés progressivement ;
- le **moteur de conseil** synthétise les résultats.

Au départ, les prédictions sont simulées pour valider le parcours utilisateur. Ensuite, chaque simulation sera remplacée par un vrai modèle.

## Architecture

```text
fit-outfit-advisor/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── services/
│   │   ├── image_service.py
│   │   ├── fit_service.py
│   │   ├── outfit_service.py
│   │   └── advice_service.py
│   │
│   ├── preprocessing/
│   │   ├── image_preprocessing.py
│   │   └── tabular_preprocessing.py
│   │
│   ├── mappings/
│   │   ├── category_mapping.py
│   │   └── color_mapping.py
│   │
│   ├── models/
│   │   ├── load_image_model.py
│   │   └── load_fit_model.py
│   │
│   └── schemas/
│       └── prediction_schemas.py
│
├── models/
│   ├── fashion_model.keras       # à générer après entraînement
│   ├── fit_model.keras           # à générer après entraînement
│   └── encoders/                 # encodeurs/scalers sauvegardés
│
├── notebooks/
│   ├── 01_train_fit_model_colab.ipynb
│   ├── 02_train_fashion_model_colab.ipynb
│   └── 03_polyvore_exploration_colab.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Pour lancer uniquement le prototype Streamlit simulé sans TensorFlow :

```bash
pip install -r requirements-streamlit.txt
```

## Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

## Structure du projet

```text
app/                 Interface Streamlit
src/services/        Services metier appelables plus tard par une API
src/preprocessing/   Preprocessing image et tabulaire
src/mappings/        Vocabulaire commun categories/couleurs
src/models/          Loaders de modeles et artefacts
src/training/        Scripts reproductibles d'entrainement
models/              Artefacts generes hors Git
data/raw/            Datasets locaux non versionnes
tests/               Tests unitaires MVP
notebooks/           Bases Colab/Kaggle
```

## Etat actuel des modeles

- Image : `predict_image()` fonctionne en simulation si `models/fashion_model.keras` est absent.
- Fit : `predict_fit()` tente les artefacts ModCloth reels puis revient au fallback simule si un fichier manque.
- Outfit : recommandations simples par regles et mappings.
- Advice : conseil final interpretable avec avertissements si confiance faible.

Artefacts fit attendus apres entrainement :

```text
models/fit_model.keras
models/encoders/fit_preprocessor.joblib
models/encoders/fit_label_encoder.joblib
models/encoders/fit_metadata.json
```

## Datasets

Place les datasets localement, sans les committer :

```text
data/raw/modcloth_final_data.json
data/raw/fashion-product-images-small/
```

Le script ModCloth accepte aussi `--dataset chemin/vers/fichier.csv` ou `.json/.jsonl`.

## Entrainement ModCloth

Le pipeline de base est disponible dans :

```bash
python -m src.training.train_fit_model --dataset data/raw/modcloth_final_data.json
```

Si le dataset est absent, le script affiche les colonnes attendues. Pour verifier uniquement le pipeline avec un mini-jeu artificiel :

```bash
python -m src.training.train_fit_model --sample --epochs 1
```

Ce mode `--sample` ne produit pas de resultats exploitables pour le rapport.

## Tests

```bash
python -m compileall app src tests
pytest
```

## Ordre de développement

1. Valider le prototype Streamlit simulé.
2. Entraîner le modèle ModCloth pour la prédiction `small / fit / large`.
3. Sauvegarder le modèle, le scaler et les encodeurs.
4. Brancher le modèle ModCloth dans `fit_service.py`.
5. Entraîner le CNN Fashion Product Images Small sur 5 à 8 classes.
6. Brancher le CNN dans `image_service.py`.
7. Améliorer le module outfit avec des règles Polyvore simplifiées.
8. Enrichir le conseil final.

## Critère de réussite de la première semaine

Le projet est réussi pour la première semaine si :

- l'application Streamlit se lance ;
- l'utilisateur peut charger une image ;
- l'utilisateur peut saisir son profil ;
- les services simulés retournent des résultats ;
- un conseil final est généré ;
- la logique métier est bien séparée de l'interface.
