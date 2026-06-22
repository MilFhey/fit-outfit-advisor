# MODCLOTH_BASELINE_ANALYSIS

## Diagnostic du baseline v1
- Le premier entrainement ModCloth a fonctionne techniquement, mais le modele n'est pas exploitable.
- Accuracy test observee : `0.6857`.
- Recalls observes :
  - `fit` : `1.00`
  - `large` : `0.00`
  - `small` : `0.00`
- La matrice de confusion indique que le modele predit presque toujours `fit`.
- Decision : ne pas promouvoir ce modele vers Streamlit.
- Conservation : les artefacts existants doivent rester comme baseline v1 et ne doivent pas etre ecrases.

## Distribution des classes test
| Classe | Effectif |
| --- | ---: |
| `fit` | 8366 |
| `large` | 1921 |
| `small` | 1910 |
| Total | 12197 |

## Accuracy vs baseline majoritaire
- Baseline majoritaire test = toujours predire `fit`.
- Accuracy baseline majoritaire = `8366 / 12197 = 0.686`.
- Accuracy modele observee = `0.6857`.
- Conclusion : l'accuracy du modele est pratiquement identique au baseline majoritaire.
- Macro F1 attendu du baseline majoritaire : environ `0.27`, car `large` et `small` ont F1 = `0`.

## Causes probables
- Des classes fortement desequilibrees, avec `fit` majoritaire.
- Optimisation/evaluation trop centree sur l'accuracy.
- Absence d'experience comparative avec baseline majoritaire et `class_weight`.
- Variables peu coherentes pour l'inference future dans le pipeline v1 :
  - `weight_kg` force a `0.0` quand la colonne source est absente ou inexploitable ;
  - `brand`, `color`, `usual_size` absents du dataset ModCloth reel et donc appris comme placeholders ou valeurs inconnues ;
  - `category` contient des valeurs natives ModCloth comme `new`, qui ne correspondent pas directement aux categories communes du produit.
- Preprocessing insuffisamment diagnostique avant entrainement.

## Analyse du warning DtypeWarning
- Warning observe : `DtypeWarning: Columns (8) have mixed types.`
- Pandas indexe cette indication a partir de 0.
- Le dataset n'est pas present dans le repo local ; le vrai nom de la colonne 8 ne peut donc pas etre prouve localement sans inventer.
- Correction ajoutee : `src.training.train_fit_model.inspect_warning_column()` lit l'en-tete CSV, affiche :
  - l'index signale ;
  - le vrai nom de colonne ;
  - les types Python observes ;
  - des exemples de valeurs ;
  - la strategie de nettoyage.
- Strategie retenue par defaut : exclure cette colonne des features V2 sauf justification metier explicite et parser dedie.

## Analyse de `weight_kg = 0.0`
- `weight_kg = 0.0` indique une valeur manquante remplacee par un zero silencieux.
- C'est incoherent metier : un poids a `0 kg` n'est pas une valeur plausible pour l'utilisateur.
- Correction V2 :
  - `weight_kg` est retire des features ModCloth V2 ;
  - aucune valeur manquante numerique n'est remplacee par `0` sans indicateur ;
  - les valeurs numeriques retenues sont imputees par mediane uniquement dans le preprocessor ajuste sur le train.

## Analyse de `category = "new"`
- `new` est une valeur native possible de la colonne `category` ModCloth.
- Elle ne doit pas etre interpretee comme la categorie commune produit (`top`, `bottom`, `dress`, etc.).
- Correction V2 :
  - `category` reste une variable dataset-native si elle existe ;
  - sa semantique est documentee dans les metadonnees ;
  - elle n'est pas mappee artificiellement vers une categorie produit.

## Plan de correction
1. Ajouter un diagnostic reproductible avant entrainement.
2. Identifier dynamiquement la colonne du `DtypeWarning`.
3. Selectionner uniquement des features qui existent vraiment dans ModCloth.
4. Retirer les champs non appris ou incoherents : `weight_kg`, `usual_size`, `brand`, `color`.
5. Split train / validation / test avant tout `fit` du preprocessor.
6. Ajuster imputer, scaler et encoder uniquement sur le train.
7. Comparer :
   - baseline majoritaire ;
   - MLP sans ponderation ;
   - MLP avec `class_weight` calcule sur le train.
8. Ajouter `EarlyStopping(restore_best_weights=True)`.
9. Sauvegarder les artefacts V2 dans `models/fit_v2/`.
10. Ne promouvoir vers Streamlit que si les criteres metier/minorites sont atteints.

## Critères de succès du second entrainement
- Recall `small` > 0.
- Recall `large` > 0.
- Macro F1 significativement superieur au baseline actuel, environ `0.27`.
- Balanced accuracy superieure au baseline majoritaire.
- Matrices de confusion brute et normalisee disponibles.
- Courbe d'apprentissage disponible.
- `metrics.json` et `metadata.json` generes.
- Aucun champ d'inference incoherent du type `weight_kg = 0.0` pour une valeur absente.

## Decision actuelle
- Modele baseline v1 : non promouvable vers Streamlit.
- Modele v2 : a entrainer dans Colab avec le dataset reel.
- Integration Streamlit du modele reel : hors perimetre tant que `models/fit_v2/metrics.json` ne valide pas les criteres ci-dessus.
