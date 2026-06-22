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

### Statut ModCloth

Le premier entrainement ModCloth a produit un modele baseline techniquement valide mais non exploitable :

- test accuracy observee : `0.6857` ;
- recall `fit` : `1.00` ;
- recall `large` : `0.00` ;
- recall `small` : `0.00`.

Ce modele est equivalent au baseline majoritaire qui predit presque toujours `fit`. Il ne doit pas etre promu vers Streamlit.

Analyse detaillee :

```text
docs/working/MODCLOTH_BASELINE_ANALYSIS.md
```

Les artefacts baseline existants doivent etre conserves comme `baseline_v1` et ne pas etre ecrases :

```text
models/fit_model.keras
models/encoders/fit_preprocessor.joblib
models/encoders/fit_label_encoder.joblib
models/encoders/fit_metadata.json
```

Le second entrainement ecrit dans une version separee :

```text
models/fit_v2/
├── fit_model.keras
├── fit_preprocessor.joblib
├── fit_label_encoder.joblib
├── metadata.json
├── metrics.json
├── confusion_matrix_raw.png
├── confusion_matrix_normalized.png
└── training_history.png
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

Le script V2 affiche avant entrainement :

- chemin et shape du dataset ;
- colonnes disponibles ;
- vraie colonne correspondant a `DtypeWarning: Columns (8)` si le dataset est CSV ;
- valeurs manquantes ;
- distribution des classes avant/apres nettoyage ;
- colonnes retenues ;
- mapping `label -> index`.

Il compare ensuite :

- baseline majoritaire ;
- MLP sans ponderation ;
- MLP avec `class_weight` calcule sur le train.

La selection entre le MLP sans ponderation et le MLP avec `class_weight` se fait exclusivement sur le jeu de validation. Le jeu de test n'est transforme et predit qu'apres selection finale, pour produire les metriques finales et les matrices de confusion finales.

Les metriques prioritaires sont :

- macro F1 ;
- balanced accuracy ;
- recall `small` ;
- recall `large`.

Si le dataset est absent, le script affiche une erreur claire. Pour verifier uniquement le pipeline avec un mini-jeu artificiel :

```bash
python -m src.training.train_fit_model --sample --epochs 1
```

Ce mode `--sample` ne produit pas de resultats exploitables pour le rapport.

### Notebook Google Colab ModCloth

Le notebook a executer dans Colab est :

```text
notebooks/01_train_fit_model_colab.ipynb
```

Il couvre uniquement le pipeline ModCloth V2 : montage Drive, mise a jour du repo GitHub, installation des dependances, telechargement Kaggle, inspection du dataset, comparaison baseline/MLP, appel de `src.training.train_fit_model`, verification des artefacts versionnes et copie vers Google Drive.

Avant execution dans Colab :

1. Cree un secret Colab `KAGGLE_API_TOKEN`.
2. Mets comme valeur soit le JSON Kaggle complet :

```json
{"username":"ton_user_kaggle","key":"ta_cle_kaggle"}
```

ou le format :

```text
ton_user_kaggle:ta_cle_kaggle
```

3. Renseigne l'URL GitHub du repo dans la cellule `REPO_URL`, ou cree un secret Colab `FIT_OUTFIT_REPO_URL` contenant l'URL du repo, par exemple :

```text
https://github.com/<ton-compte-github>/fit-outfit-advisor.git
```

Si la branche par defaut du repo n'est pas `main`, modifie aussi la variable `BRANCH` dans la meme cellule.

4. Lance les cellules dans l'ordre.

Le notebook telecharge par defaut le dataset Kaggle :

```text
rmisra/clothing-fit-dataset-for-size-recommendation
```

Si Kaggle fournit ModCloth en JSON/JSONL, le notebook le convertit en CSV temporaire dans `/content/fit-outfit-runtime/data/modcloth_final_data.csv` avant d'appeler le script d'entrainement.

Artefacts generes dans le repo Colab :

```text
models/fit_v2/fit_model.keras
models/fit_v2/fit_preprocessor.joblib
models/fit_v2/fit_label_encoder.joblib
models/fit_v2/metadata.json
models/fit_v2/metrics.json
models/fit_v2/confusion_matrix_raw.png
models/fit_v2/confusion_matrix_normalized.png
models/fit_v2/training_history.png
```

`models/fit_v2/metrics.json` separe :

- `validation_metrics` ;
- `test_metrics` ;
- `selected_experiment` ;
- `reason_for_selection` ;
- `feature_columns` ;
- `dataset_row_counts` ;
- `class_distribution_train` ;
- `class_distribution_validation` ;
- `class_distribution_test`.

Copie finale vers Google Drive :

```text
/content/drive/MyDrive/fit-outfit-advisor/artifacts/modcloth_fit_v2/
```

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
