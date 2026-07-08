# CODEX_AUDIT - Fit & Outfit Advisor

## Etat actuel du repo
- Repo local inspecte dans `fit-outfit-advisor/`.
- Pas de depot Git detecte depuis ce dossier au moment de l'audit.
- Source officielle actuelle : `docs/working/PROJECT_CONTEXT.md`.
- Ancien emplacement `docs/PROJECT_CONTEXT.md` absent dans l'etat post-audit.
- Prototype Streamlit present et fonctionnel en mode simule.
- Services separes dans `src/services/`, mappings dans `src/mappings/`, preprocessing dans `src/preprocessing/`, loaders dans `src/models/`.
- Notebooks presents mais encore au stade TODO.

## Fichiers presents utiles
- `app/streamlit_app.py`
- `src/services/{image_service.py,fit_service.py,outfit_service.py,advice_service.py}`
- `src/preprocessing/{image_preprocessing.py,tabular_preprocessing.py}`
- `src/mappings/{category_mapping.py,color_mapping.py}`
- `src/models/{load_image_model.py,load_fit_model.py}`
- `src/schemas/prediction_schemas.py`
- `tests/test_services.py`
- `notebooks/{01_train_fit_model_colab.ipynb,02_train_fashion_model_colab.ipynb,03_polyvore_exploration_colab.ipynb}`
- `models/README.md`
- `requirements.txt`, `requirements-streamlit.txt`

## Coherences avec PROJECT_CONTEXT.md
- Architecture modulaire deja amorcee.
- Streamlit appelle les services au lieu de porter toute la logique metier.
- Simulation acceptee pour stabiliser le parcours MVP.
- Mappings categorie/couleur existants.
- Loaders modeles retournant `None` si artefact absent.
- Dossiers notebooks et models conformes a la cible V0.

## Ecarts ou risques
- Chemins modeles dupliques et relatifs au dossier d'execution.
- `predict_fit(use_real_model=True)` levait `NotImplementedError` apres chargement potentiel du modele.
- Service fit pas encore pret a charger scaler/encodeurs/metadonnees.
- Conseil final retournait une string seule, moins conforme au format `FinalAdvice` documente.
- Streamlit affichait surtout des JSON bruts, peu lisibles pour une demo.
- Tests trop limites : pas de couverture mappings, fallbacks, cles attendues.
- Notebook ModCloth non executable comme pipeline reproductible.

## Priorites de developpement
1. Stabiliser l'affichage Streamlit demo.
2. Prepararer l'inference ModCloth avec fallback sans crash.
3. Centraliser les chemins d'artefacts.
4. Ajouter une base d'entrainement ModCloth reproductible.
5. Renforcer les tests unitaires sans modele reel.
6. Documenter l'etat exact des modeles et datasets.

## Actions realisees par Codex
- Creation de `docs/working/CODEX_AUDIT.md`.
- Ajout d'une configuration de chemins portable.
- Renforcement du preprocessing tabulaire ModCloth.
- Ajout d'un loader d'artefacts fit modele + preprocessor + label encoder + metadata.
- Adaptation de `fit_service` pour tenter le modele reel si complet et revenir a la simulation sinon.
- Structuration du conseil final sous forme de dict testable.
- Amelioration de l'interface Streamlit avec affichage principal lisible et expander technique.
- Ajout d'un script d'entrainement ModCloth avec erreur claire si dataset absent.
- Renforcement des tests MVP.
- Mise a jour README.

## Stabilisation post-audit
- Etat final des fichiers de contexte :
  - `docs/working/PROJECT_CONTEXT.md` est present et considere comme source officielle.
  - `docs/PROJECT_CONTEXT.md` n'est pas present dans l'etat actuel du repo.
- Commandes executees :
  - `python -m compileall app src tests`
  - `pytest` avec `.venv/Scripts` ajoute au `PATH` de la session PowerShell.
- Resultat `compileall` :
  - OK, compilation de `app`, `src` et `tests` sans erreur.
- Resultat `pytest` :
  - OK, 9 tests passes.
- Corrections effectuees :
  - `predict_image(..., use_real_model=True)` revient maintenant en simulation si `models/fashion_active/` est absent ou non promu.
  - Ajout d'un test de fallback image en mode reel demande.
  - Ajout d'un test de chemins centralises et loaders absents sans crash.
  - Ajout de `tests/conftest.py` pour rendre les imports `src.*` stables sous `pytest`.
- Limites restantes :
  - Les artefacts reels ModCloth et image ne sont pas encore presents.
  - Le fallback image ne remplace pas encore un vrai pipeline CNN ; il preserve seulement le parcours MVP.
  - Les notebooks restent des bases de travail, le script `src/training/train_fit_model.py` est le pipeline le plus concret pour ModCloth.

## Analyse V2 ModCloth et plan V3
- Decision V2 :
  - V2 ameliore le baseline majoritaire mais reste non fiable pour une recommandation ferme de taille.
  - V2 est conserve comme resultat academique et baseline ameliore.
  - V2 ne doit pas etre promu vers Streamlit comme conseil utilisateur.
- Corrections de garde-fou :
  - `metadata.json` genere par `src/training/train_fit_model.py` force `promotable_to_streamlit: false`.
  - `metadata.json` ajoute `model_status: "experimental_only"`.
  - `fit_service.py` retourne `uncertain` si un artefact charge n'est pas explicitement promu ou si la confiance est sous le seuil d'abstention.
  - Les chemins fit actifs centralises pointent vers `models/fit_active/`, pas vers `models/fit_v2/`.
  - `models/fit_v2/` reste un emplacement experimental et non actif par defaut.
  - Le contrat d'inference est construit depuis les `feature_columns`; `body_type` n'apparait que si la colonne fait reellement partie des features apprises.
  - Les categories vestimentaires explicites et categories commerciales ambigues sont separees dans le preprocessing.
  - `height_cm_missing` est ajoute comme indicateur de mensuration manquante pour la taille.
- Nouveau document :
  - `docs/working/MODCLOTH_V3_EXPERIMENT_PLAN.md`.
- Limites restantes :
  - Le dataset reel et les artefacts V2 ne sont pas presents localement ; les distributions et performances V3 reelles doivent etre calculees dans Colab ou dans un environnement disposant du dataset.
  - Les mensurations `bust`, `hips`, `waist`, `bra size`, `cup size` restent candidates V3 et ne sont pas encore integrees au pipeline.

## Implementation pipeline V3 experimental
- Ajout de `src/training/train_fit_model_v3.py`.
- Ajout du preprocessing V3 dans `src/preprocessing/tabular_preprocessing.py` :
  - `height` parse en `height_cm` ;
  - outliers `height_cm` hors `[130, 210]` convertis en valeurs manquantes ;
  - `hips`, `bra_size`, `cup_size` retenues ;
  - indicateurs `height_cm_missing`, `hips_missing`, `bra_size_missing`, `cup_size_missing` ;
  - exclusion des colonnes post-achat, identifiants et mensurations trop manquantes.
- Comparaisons V3 implementees :
  - baseline majoritaire ;
  - regression logistique ;
  - MLP TensorFlow ;
  - MLP TensorFlow avec `class_weight`.
- Selection du modele V3 exclusivement sur validation.
- Test utilise apres selection finale uniquement.
- Artefacts V3 ecrits dans `models/fit_v3/`.
- Garde-fous V3 :
  - `model_status: "experimental_only"` ;
  - `promotable_to_streamlit: false` ;
  - aucune ecriture vers `models/fit_active/`.
- Controle local :
  - `python -m compileall app src tests` OK ;
  - `pytest` OK, 20 tests passes ;
  - invocation locale du script V3 non executee jusqu'au bout car l'environnement Python actif ne contient pas `scikit-learn`; le lancement cible reste Colab apres installation de `requirements.txt`.

## Cloture ModCloth et priorite Fashion CNN V0
- Decision :
  - ModCloth V3 est conserve comme experimentation academique et baseline amelioree.
  - Aucun artefact ModCloth n'est promu vers `models/fit_active/`.
  - La priorite active devient Fashion Product Images Small pour la reconnaissance image.
- Nouveau document :
  - `docs/working/FASHION_CNN_V0_PLAN.md`.
- Configuration ajoutee :
  - `config/fashion_v1_classes.json`.
  - Le mapping `product_type_v0 -> articleType acceptes` est valide apres audit Colab.
- Cible image V0 :
  - `product_type_v0`, derivee de `styles.csv.articleType`.
  - `canonical_category` est derivee ensuite pour le moteur outfit.
  - Classes detaillees validees V1.1 : `tshirt`, `shirt`, `top`, `jeans`, `trousers`, `shorts`, `dress`, `outerwear`, `casual_shoes`, `sports_shoes`, `dress_shoes`, `sandals`, `flip_flops`, `heels`, `bag`, `watch`, `sunglasses`, `wallet`, `belt`, `jewellery`.
  - Decision V1.1 : `Flats` n'est plus une sortie visible du CNN ; `articleType = "Flats"` est conserve et mappe vers `dress_shoes`.
  - Resultats experimentaux V1.1 : MobileNetV2, accuracy test `0.8748`, balanced accuracy test `0.8483`, macro F1 test `0.8526`, weighted F1 test `0.8725`.
  - Classe fragile restante : `dress_shoes`, precision `0.729`, recall `0.456`, F1 `0.561`.
  - Seuil minimal retenu : `450` images lisibles par classe.
  - `cap` exclu de V1 : `283` images lisibles seulement.
  - Categories canoniques derivees : `top`, `bottom`, `dress`, `shoes`, `outerwear`, `bag`, `accessory`.
- Garde-fous image :
  - `models/fashion_v1/` est experimental.
  - `models/fashion_active/` sera le seul emplacement actif futur.
  - Le service image doit refuser tout artefact non promu et revenir au fallback simule.
- Notebook cible :
  - `notebooks/02_train_fashion_model_colab.ipynb` doit inspecter le dataset, produire le tableau final par `articleType`, puis s'arreter avant entrainement si le mapping reste en brouillon.
- Entrainement Fashion V1 :
  - `src/training/train_fashion_model_v1.py` implemente le dry-run, le split stratifie, `simple_cnn`, `mobilenet_v2`, la selection validation-only et l'evaluation test finale.
  - Les artefacts sont ecrits dans `models/fashion_v1/` avec `model_status: "experimental_only"` et `promotable_to_streamlit: false`.
- Analyse post-entrainement Fashion V1.1 :
  - Ajout de `src/analysis/analyze_fashion_v1_abstention.py`.
  - Le seuil de confiance est selectionne sur validation uniquement.
  - Le test est evalue une seule fois au seuil retenu ou diagnostique.
  - Sous seuil, la sortie future doit etre `unknown`, pas une classe ferme.
- Promotion controlee Fashion V1.1 :
  - Seuil retenu : `0.90`.
  - Test au seuil : coverage `0.7083`, unknown rate `0.2917`, accuracy non-unknown `0.9695`, macro F1 non-unknown `0.9425`.
  - Artefacts copies localement dans `models/fashion_active/`.
  - Metadata active : `model_status: "promoted"`, `promotable_to_streamlit: true`, `abstention_strategy.minimum_confidence: 0.90`.
  - Smoke test local du chargement via `image_service.predict_image(..., use_real_model=True)` OK.

## Priorite Outfit Compatibility V0
- Ajout du plan `docs/working/OUTFIT_COMPATIBILITY_V0_PLAN.md`.
- Ajout de `config/outfit_v1_config.json` en statut `draft_requires_dataset_inspection`.
- Ajout des helpers de mapping Polyvore -> Fashion V1 et preprocessing paires outfit.
- `outfit_service` conserve son contrat historique et ajoute les champs V0 : `input_product_type`, `recommended_product_types`, `compatible_roles`, `raw_compatibility_score`, `model_status`.
- Notebook Polyvore ajuste pour utiliser Hugging Face `mvasil/polyvore-outfits` avec le secret Colab `HUGGIN_KEY`, inspecter `disjoint`/`nondisjoint` si disponibles et sauvegarder le dataset dans Google Drive.
- Aucun entrainement Polyvore n'est lance dans cette etape.
