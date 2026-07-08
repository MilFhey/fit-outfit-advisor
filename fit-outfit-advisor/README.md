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
│   ├── fashion_v1/               # artefacts image expérimentaux
│   ├── fashion_active/           # artefacts image explicitement promus uniquement
│   ├── fit_v2/                   # artefacts ModCloth V2 expérimentaux
│   ├── fit_v3/                   # artefacts ModCloth V3 expérimentaux
│   └── fit_active/               # artefacts fit explicitement promus uniquement
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

- Image : `predict_image()` fonctionne en simulation si `models/fashion_active/` est absent ou non promu.
- Fit : `predict_fit()` tente uniquement les artefacts ModCloth explicitement promus dans `models/fit_active/`, puis revient au fallback simule si un fichier manque ou si les metadata ne sont pas promues.
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

Decision actuelle V2 :

- V2 ameliore le baseline majoritaire (`macro F1 test 0.357` vs `0.271`, `balanced accuracy test 0.434` vs `0.333`).
- V2 reste insuffisant pour un conseil utilisateur fiable (`accuracy 0.385`, precision faible sur `small` et `large`, recall `fit 0.341`).
- Les artefacts V2 doivent etre marques `promotable_to_streamlit: false` et `model_status: "experimental_only"`.
- `models/fit_v2/` n'est pas un emplacement actif par defaut.
- Aucun modele ne devient actif uniquement parce qu'il existe dans `models/`.
- Un modele fit est utilisable par le service seulement s'il est place dans `models/fit_active/` avec `model_status: "promoted"` et `promotable_to_streamlit: true`.
- Tout autre etat (`experimental_only`, metadata absente, champ absent, metadata illisible) est refuse en fail-closed et mene a `uncertain` ou au fallback simule.
- L'analyse V3 est planifiee dans `docs/working/MODCLOTH_V3_EXPERIMENT_PLAN.md` avant toute promotion vers Streamlit.

Le pipeline V3 est separe et experimental. Il ajoute les mensurations pre-achat retenues (`height`, `hips`, `bra size`, `cup size`) avec indicateurs de valeurs manquantes, compare baseline majoritaire, regression logistique, MLP et MLP pondere, puis ecrit uniquement dans :

```text
models/fit_v3/
├── fit_model.keras               # seulement si un MLP est selectionne
├── fit_estimator.joblib          # seulement si la regression logistique est selectionnee
├── fit_preprocessor.joblib
├── fit_label_encoder.joblib
├── metadata.json
├── metrics.json
├── confusion_matrix_raw.png
├── confusion_matrix_normalized.png
└── training_history.png
```

Commande V3 :

```bash
python -m src.training.train_fit_model_v3 --dataset data/raw/modcloth_final_data.json --epochs 30 --batch-size 64
```

Pour entrainer uniquement sur les categories vestimentaires explicites :

```bash
python -m src.training.train_fit_model_v3 --dataset data/raw/modcloth_final_data.json --category-scope explicit
```

V3 reste toujours `model_status: "experimental_only"` et `promotable_to_streamlit: false`. Ne copie pas ces artefacts vers `models/fit_active/`.

### Analyse d'abstention V3

Une fois les artefacts V3 archives localement dans `models/fit_v3_all/` et `models/fit_v3_explicit/`, analyse les seuils de confiance sans reentrainer :

```bash
python -m src.analysis.analyze_fit_v3_abstention --dataset data/raw/modcloth_final_data.json --run both
```

Le script teste les seuils de `0.35` a `0.90`, selectionne un seuil sur validation uniquement, puis evalue le test une seule fois au seuil retenu ou diagnostique. Rapports attendus :

```text
reports/modcloth_v3_abstention_all.json
reports/modcloth_v3_abstention_explicit.json
```

Si aucun seuil n'atteint `coverage >= 25%` et precision `small`/`large >= 0.40`, la conclusion reste : pas de recommandation ferme, utiliser `uncertain`.

Resultat local actuel :

- `fit_v3_all` : aucun seuil acceptable ; meilleur seuil diagnostique `0.60`, coverage test `0.36%`, abstention test `99.64%`.
- `fit_v3_explicit` : aucun seuil acceptable ; meilleur seuil diagnostique `0.45`, coverage test `3.35%`, abstention test `96.65%`.
- Decision : V3 reste academique et `experimental_only`, sans promotion vers `models/fit_active/`.

Le module ModCloth est donc cloture comme experimentation academique pour le MVP actuel. La priorite de developpement passe au modele image Fashion Product Images Small.

## Fashion CNN V0

Le plan de travail image est documente dans :

```text
docs/working/FASHION_CNN_V0_PLAN.md
```

La cible apprise par le futur CNN sera `product_type_v0`, derivee de la colonne source `articleType` de `styles.csv`.
`canonical_category` reste un role metier derive apres prediction pour le moteur outfit.

Exemple attendu apres validation du mapping :

```text
articleType = "Tshirts" -> product_type_v0 = "tshirt" -> canonical_category = "top"
```

Les classes `product_type_v0` candidates sont :

```text
tshirt, shirt, top, jeans, trousers, shorts, dress, outerwear,
casual_shoes, sports_shoes, dress_shoes, sandals, flip_flops,
heels, bag, watch, sunglasses, wallet, belt, jewellery
```

Le mapping explicite est dans :

```text
config/fashion_v1_classes.json
```

Ce fichier est valide pour l'entrainement V1 apres audit Colab. Le seuil minimal retenu est `450` images lisibles par classe. `cap` est exclu de la V1 car la classe ne contient que `283` images lisibles.

Decision V1.1 apres analyse du premier entrainement : `Flats` n'est plus une sortie visible du CNN. Les images `articleType = "Flats"` restent utilisees comme exemples de `product_type_v0 = "dress_shoes"`, car la classe separee `flats` etait trop mal reconnue (`F1 test 0.138`, recall `0.080`) et etait majoritairement confondue avec `heels`.

Resultat experimental V1.1 MobileNetV2 :

- accuracy test : `0.8748` ;
- balanced accuracy test : `0.8483` ;
- macro F1 test : `0.8526` ;
- weighted F1 test : `0.8725`.

`dress_shoes` reste la classe la plus fragile (`precision 0.729`, `recall 0.456`, `F1 0.561`). Avant toute promotion, il faut generer l'analyse de seuils de confiance :

```bash
python -m src.analysis.analyze_fashion_v1_abstention \
  --metadata-csv data/raw/fashion-product-images-small/styles.csv \
  --image-dir data/raw/fashion-product-images-small/images \
  --artifact-dir models/fashion_v1 \
  --output reports/fashion_v1_abstention.json
```

Le seuil est choisi sur validation uniquement. Le test est evalue une seule fois au seuil retenu ou diagnostique. Si un modele image est promu plus tard, le seuil retenu devra etre inscrit dans `metadata.json` sous `abstention_strategy.minimum_confidence`.

Decision de promotion controlee :

- Fashion V1.1 est promu localement dans `models/fashion_active/` ;
- metadata active : `model_status: "promoted"` et `promotable_to_streamlit: true` ;
- seuil actif : `abstention_strategy.minimum_confidence: 0.90` ;
- sous seuil, `image_service` retourne `product_type: "unknown"` ;
- smoke test local `predict_image(..., use_real_model=True)` OK avec chargement du modele actif.

Le pipeline impose est :

```text
styles.csv
-> mapping articleType vers product_type_v0
-> mapping deterministe product_type_v0 vers canonical_category
-> exclusion labels non retenus
-> verification fichier image present et lisible
-> comptage final par classe
-> seuil minimal documente par classe
-> split stratifie
```

Le notebook d'inspection image est :

```text
notebooks/02_train_fashion_model_colab.ipynb
```

Il peut etre execute pour verifier le dataset et regenerer le rapport d'audit. L'entrainement image reste dans une etape separee.

Artefacts futurs attendus, sans promotion automatique :

```text
models/fashion_v1/
├── fashion_model.keras
├── label_encoder.joblib
├── metadata.json
├── metrics.json
├── confusion_matrix_raw.png
├── confusion_matrix_normalized.png
├── training_history.png
└── sample_predictions.png
```

`models/fashion_v1/` reste l'emplacement experimental. Le service image utilise uniquement `models/fashion_active/` si les metadata contiennent `model_status: "promoted"` et `promotable_to_streamlit: true`. Sa sortie minimale est `product_type`, `canonical_category`, `confidence` et `model_status`.

Commande de diagnostic sans entrainement :

```bash
python -m src.training.train_fashion_model_v1 \
  --metadata-csv data/raw/fashion-product-images-small/styles.csv \
  --image-dir data/raw/fashion-product-images-small/images \
  --dry-run
```

Commande d'entrainement experimental Fashion V1 :

```bash
python -m src.training.train_fashion_model_v1 \
  --metadata-csv data/raw/fashion-product-images-small/styles.csv \
  --image-dir data/raw/fashion-product-images-small/images \
  --architecture both \
  --epochs 12 \
  --batch-size 32 \
  --image-size 224
```

Le script compare `simple_cnn` et `mobilenet_v2`, selectionne uniquement sur validation, puis evalue le test une seule fois apres selection. Les artefacts restent `experimental_only`.

Analyse des seuils de confiance apres entrainement :

```bash
python -m src.analysis.analyze_fashion_v1_abstention \
  --metadata-csv data/raw/fashion-product-images-small/styles.csv \
  --image-dir data/raw/fashion-product-images-small/images \
  --artifact-dir models/fashion_v1 \
  --output reports/fashion_v1_abstention.json
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

1. Cree un secret Colab `KAGGLE_API`.
2. Mets comme valeur le token Kaggle API commencant par `KGAT_`.
3. Si le repo GitHub est prive, cree aussi un secret optionnel `GITHUB_TOKEN`.
4. Verifie la variable `REPO_URL` dans le notebook.
5. Si la branche par defaut du repo n'est pas `main`, modifie aussi la variable `BRANCH`.
6. Lance les cellules dans l'ordre.

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
5. Cloturer ModCloth comme experimentation academique non promue.
6. Inspecter Fashion Product Images Small dans Colab et valider les classes V0. Fait pour Fashion V1.
7. Entraîner le CNN Fashion Product Images Small sur `product_type_v0`, puis mapper vers `canonical_category`.
8. Brancher le CNN promu dans `image_service.py`.
9. Améliorer le module outfit avec audit Polyvore, baseline cooccurrence et règles alignées Fashion V1.
10. Enrichir le conseil final.

## Outfit Compatibility V0

Le prochain module est documente dans :

```text
docs/working/OUTFIT_COMPATIBILITY_V0_PLAN.md
```

Objectif : recommander des `product_type_v0` complementaires, pas generer une tenue complete. Le module doit rester aligne sur Fashion V1.1 via `product_type_v0`, `canonical_category` et `outfit_role`.

Livrables initiaux :

```text
config/outfit_v1_config.json
notebooks/03_polyvore_exploration_colab.ipynb
reports/polyvore_v0_dataset_audit.json
```

Aucun entrainement Polyvore ne doit etre lance avant audit dataset. `models/outfit_v1/` restera experimental et `models/outfit_active/` sera le seul emplacement actif futur avec metadata promues.

Le notebook Polyvore utilise comme source principale Hugging Face `mvasil/polyvore-outfits`, via le secret Colab `HUGGIN_KEY`. Il inspecte les configurations/splits `disjoint` et `nondisjoint` si disponibles, puis sauvegarde une copie du dataset dans Google Drive sous `MyDrive/fit-outfit-advisor/datasets/mvasil_polyvore_outfits`.

## Critère de réussite de la première semaine

Le projet est réussi pour la première semaine si :

- l'application Streamlit se lance ;
- l'utilisateur peut charger une image ;
- l'utilisateur peut saisir son profil ;
- les services simulés retournent des résultats ;
- un conseil final est généré ;
- la logique métier est bien séparée de l'interface.
