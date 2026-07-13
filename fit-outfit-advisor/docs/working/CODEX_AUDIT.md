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
- Ajout de la synthese rapport `docs/working/PROJECT_REPORT_SYNTHESIS.md`.
- Ajout initial de `config/outfit_v1_config.json` en statut `draft_requires_dataset_inspection`, puis promotion en `validated_for_baseline_v0` apres audit schema/mapping Colab.
- Ajout des helpers de mapping Polyvore -> Fashion V1 et preprocessing paires outfit.
- `outfit_service` conserve son contrat historique et ajoute les champs V0 : `input_product_type`, `recommended_product_types`, `compatible_roles`, `raw_compatibility_score`, `model_status`.
- Notebook Polyvore ajuste pour utiliser Hugging Face `mvasil/polyvore-outfits` avec le secret Colab `HUGGIN_KEY`, inspecter `disjoint`/`nondisjoint` si disponibles et sauvegarder le dataset dans Google Drive.
- Notebook Polyvore ajuste pour verifier d'abord la copie Drive et eviter un retelechargement systematique du dataset.
- Rapport audit recu le 2026-07-09 : le loader `datasets` expose `item_id` + `image`, mais la liste des fichiers HF contient les metadata brutes Polyvore (`categories.csv`, `polyvore_item_metadata.json`, splits JSON et fichiers compatibility). Notebook ajuste pour telecharger/cache Drive et inspecter ces fichiers raw avant baseline cooccurrence.
- Resultats Colab confirmes : loader seul non suffisant pour la baseline, baseline via fichiers raw HF possible ; split rows loader = 71 967 / 14 657 / 70 035 pour `disjoint` train/validation/test et 204 679 / 25 132 / 47 854 pour `nondisjoint`.
- Rapport raw metadata recu le 2026-07-11 : `raw_metadata_ready_for_schema_audit: true`, 29 fichiers raw, 251 008 items metadata, 68 306 outfits titres, splits disjoint/nondisjoint exploitables pour schema/mapping.
- Aucun entrainement Polyvore n'est lance dans cette etape.

## Audit schema/mapping Polyvore V0
- Ajout de `src/analysis/analyze_polyvore_v0_schema_mapping.py`.
- Ajout de `reports/polyvore_v0_schema_mapping_audit.json`.
- Mise a jour de `notebooks/03_polyvore_exploration_colab.ipynb` pour generer `reports/polyvore_v0_schema_mapping_audit.json` depuis le cache raw HF Colab/Drive avant l'arret volontaire.
- Verification : `models/polyvore/polyvore_v0_dataset_audit (2).json` est identique a `reports/polyvore_v0_dataset_audit.json`.
- Le script construit l'audit depuis les raw HF quand ils sont disponibles localement ou via `--raw-root`.
- Le rapport schema/mapping inclut le resume de l'audit dataset officiel : decision, possibilite de baseline cooccurrence, split rows du loader et fichiers raw cles.
- Champs analyses : structure `items` des splits, liaison `item_id` vers `polyvore_item_metadata.json`, distributions `semantic_category`, `category_id`, `catgeories`, `title`, `url_name`.
- Le mapping propose reste aligne avec Fashion V1.1 et ne produit que les roles `top`, `bottom`, `dress`, `shoes`, `outerwear`, `bag`, `accessory`.
- Le rapport genere localement signale que les raw HF ne sont pas dans le workspace : `raw_files_missing_requires_colab_or_drive_raw_root`.
- Decision : aucune mise a jour de `config/outfit_v1_config.json`, aucun entrainement TensorFlow et aucune integration Streamlit tant que le rapport n'a pas ete regenere avec les raw.
- Resultat Colab transmis : `schema_mapping_ready_for_manual_review`, 540 539 items lies aux metadata, 50 mappings proposes, 50 labels exclus documentes.
- Interpretation : baseline cooccurrence possible apres revue du mapping ; le loader HF seul reste insuffisant.
- Correction post-resultat : mapping explicite de `outerwear`/`hoodies`/`vests`, correction du faux positif `cap` sur `capri cropped pants`, ajout de `converse`/`chuck taylor` vers `sports_shoes`.
- Ajout d'un sanity check dans le notebook avant generation du rapport schema/mapping pour detecter une version Colab stale du script.
- Promotion manuelle d'un noyau conservateur de 28 labels dans `config/outfit_v1_config.json`.
- `validate_outfit_v1_config(..., require_ready=True)` passe maintenant sur la config officielle.
- Prochaine etape : baseline cooccurrence Polyvore, toujours sans entrainement TensorFlow.

## Baseline cooccurrence Polyvore V0
- Ajout de `src/analysis/build_polyvore_v0_cooccurrence_baseline.py`.
- Ajout de `reports/polyvore_v0_cooccurrence_baseline.json`.
- Ajout d'une cellule Colab `Baseline cooccurrence Polyvore V0` dans `notebooks/03_polyvore_exploration_colab.ipynb`.
- Le script applique le mapping valide, construit les paires positives compatibles, agrege les cooccurrences par `product_type_v0`, produit des recommandations et controle la fuite de paires positives exactes entre splits.
- Rapport local : `baseline_ready: false`, raison `raw_files_missing_requires_colab_or_drive_raw_root`.
- Aucun entrainement TensorFlow et aucune integration Streamlit.
- Resultat Colab recu : `baseline_ready: true`, 608 038 paires dirigees agregees et 32 paires `product_type_v0` uniques sur l'agregat initial.
- Correction d'interpretation : le flag de fuite initial comparait aussi `disjoint` et `nondisjoint`, qui sont deux configurations alternatives. Le rapport distingue maintenant `primary_baseline` sur `disjoint`, `aggregate_by_config` et la fuite intra-config uniquement.
- Resultat suivant : la fuite intra-config reste vraie. Correction de design : `primary_baseline` est maintenant calculee uniquement sur `disjoint_train`; l'audit expose `has_primary_train_eval_positive_pair_leakage` et bloque l'evaluation brute si des paires exactes train/eval se recouvrent.
- Dernier resultat transmis et versionne en resume dans `reports/polyvore_v0_cooccurrence_baseline.json` :
  - `baseline_ready: true` ;
  - `reason: cooccurrence_baseline_built_from_raw_metadata` ;
  - TensorFlow non utilise ;
  - config primaire : `disjoint` ;
  - split primaire : `disjoint_train` ;
  - paires dirigees split primaire : `92390` ;
  - paires `product_type_v0` uniques split primaire : `26` ;
  - fuite primaire train/eval : vraie (`disjoint train__valid = 28`, `train__test = 0`) ;
  - decision : `train_only_baseline_built_evaluation_requires_leakage_filter`.
- Le rapport local indique explicitement que le JSON complet Colab n'est pas disponible dans le workspace et que le contenu versionne est un resume transcrit depuis le resultat fourni.
- Etape suivante identifiee a ce moment : ajouter/regenerer une evaluation filtree qui retire des splits validation/test les paires positives exactes deja observees dans `train`, puis seulement ensuite envisager une integration experimentale dans `outfit_service`.
- Suite realisee localement :
  - ajout de `leakage_filtered_evaluation` dans `src/analysis/build_polyvore_v0_cooccurrence_baseline.py` ;
  - filtrage des observations validation/test dont la paire positive exacte existe deja dans `train` ;
  - calcul par split de `precision_at_k`, `recall_at_k`, `ndcg_at_k` et `mrr` sur les paires evaluables ;
  - ajout d'un indicateur `leakage_filtered_evaluation_ready` ;
  - `baseline_decision` peut maintenant passer a `train_only_baseline_ready_with_leakage_filtered_evaluation` si des paires restent evaluables apres filtrage.
- Test synthetique mis a jour : il verifie qu'une paire validation fuitee est retiree et qu'une paire validation propre reste evaluee.
- Validation locale avec `.venv311` : `python -m compileall app src tests` OK et `pytest --basetemp=.tmp_pytest -p no:cacheprovider` OK, 46 tests passes.
- Prochaine etape : regenerer `reports/polyvore_v0_cooccurrence_baseline.json` dans Colab avec les raw HF pour obtenir les vraies metriques filtrees, puis decider de l'integration experimentale.
- Rerun Colab recu apres mise a jour :
  - `baseline_ready: true` ;
  - `reason: cooccurrence_baseline_built_from_raw_metadata` ;
  - TensorFlow non utilise ;
  - config primaire : `disjoint` ;
  - split primaire : `disjoint_train` ;
  - paires dirigees split primaire : `92390` ;
  - paires `product_type_v0` uniques split primaire : `26` ;
  - fuite primaire train/eval : vraie ;
  - decision : `train_only_baseline_ready_with_leakage_filtered_evaluation`.
- Interpretation : la baseline devient candidate experimentale sous evaluation filtree, mais pas encore integree. Les metriques detaillees `leakage_filtered_evaluation` doivent etre relues depuis le JSON Drive complet ou depuis un nouveau rerun de la cellule 14.
- Notebook mis a jour : la cellule `Baseline cooccurrence Polyvore V0` affiche maintenant un tableau des metriques filtrees par split (`precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr`, comptes bruts/filtres/evaluables).
- Nouveau rerun Colab recu avec le tableau `leakage_filtered_evaluation` :
  - valid : `evaluable_directed_pair_count=16492`, `mrr=0.711036`, `recall_at_k_3=0.917596`, `ndcg_at_k_3=0.751335` ;
  - test : `evaluable_directed_pair_count=80212`, `mrr=0.709846`, `recall_at_k_3=0.929375`, `ndcg_at_k_3=0.755142`.
- Decision : metriques stables et defendables pour une integration experimentale fail-closed, sans TensorFlow et sans promotion modele.
- Integration locale : `src/services/outfit_service.py` lit `reports/polyvore_v0_cooccurrence_baseline.json` uniquement si `baseline_decision=train_only_baseline_ready_with_leakage_filtered_evaluation`, `leakage_filtered_evaluation_ready=true`, TensorFlow non utilise, et seuils minimums filtres atteints. Sinon, fallback rule-based.
- Validation locale : `python -m compileall app src tests` OK et `.venv311\Scripts\python.exe -m pytest --basetemp=.tmp_pytest -p no:cacheprovider` OK, 47 tests passes.

## Outfit Compatibility V1 TensorFlow experimental
- Clarification objectif utilisateur : l'application cible doit pouvoir evaluer une tenue et ses associations depuis des images produit, puis combiner apprentissage et regles codees, notamment pour les couleurs.
- Ajout de `src/training/train_outfit_model_v1.py`.
- Le script entraine un MLP TensorFlow binaire `compatible` / `not_compatible` sur les paires Polyvore mappees vers Fashion V1.1.
- Il conserve les garde-fous :
  - split primaire `disjoint` par outfit/set ;
  - `item_id` et `outfit_id` interdits comme features directes ;
  - paires positives exactes de train filtrees de validation/test ;
  - negatifs difficiles du meme role candidat quand possible ;
  - seuil choisi uniquement sur validation ;
  - comparaison finale au baseline cooccurrence.
- Artefacts cibles : `models/outfit_v1/outfit_model.keras`, `outfit_preprocessor.joblib`, `metadata.json`, `metrics.json`, courbes et matrices de confusion.
- Statut par defaut : `model_status: experimental_only`, `promotable_to_streamlit: false`; aucune ecriture vers `models/outfit_active/`.
- Ajout de tests unitaires pour la preparation des splits TensorFlow et la selection de seuil validation-only.
- Validation locale cible : `python -m compileall src\training\train_outfit_model_v1.py tests\test_services.py` OK ; tests ciblés Outfit TensorFlow OK.
- Resultats Colab analyses depuis `models/outfit_v1/` :
  - `beats_cooccurrence_baseline_on_test: false` ;
  - test TensorFlow MLP : accuracy `0.5008`, balanced accuracy `0.5008`, macro F1 `0.3611`, ROC AUC `0.5016` ;
  - le modele predit presque tout en `compatible` : recall `compatible` `0.9684`, recall `not_compatible` `0.0332` ;
  - loss proche de `0.693` et AUC proche de `0.50`.
- Ajout de `reports/outfit_v1_training_analysis.json`.
- Decision : `models/outfit_v1/` reste experimental_only et non promu. La baseline cooccurrence reste la voie MVP. L'experience TensorFlow est utile pour le rapport car elle demontre une tentative controlee et une decision de non-promotion justifiee.

## Outfit Compatibility V2 multimodal
- Demande utilisateur : passer a un vrai module outfit capable de recommander depuis une image, d'evaluer une tenue multi-images et de mettre le machine learning au centre.
- Ajout de `src/preprocessing/outfit_v2_features.py` :
  - extraction MobileNetV2 embeddings ;
  - extraction couleur dominante ;
  - famille couleur et harmonie couleur ;
  - features numeriques/categorielles V2 ;
  - validation explicite contre `item_id`, `outfit_id`, `set_id` comme features directes.
- Ajout de `src/training/train_outfit_model_v2.py` :
  - charge raw Polyvore + images Hugging Face ;
  - construit les paires positives/negatives depuis le pipeline V1 ;
  - ajoute image, couleur, taxonomie et cooccurrence V0 comme features ;
  - entraine un modele TensorFlow Keras binaire pairwise ;
  - selectionne le seuil sur validation ;
  - sauvegarde `metadata.json`, `metrics.json`, `product_type_prototypes.json`, courbe d'entrainement, matrice de confusion et exemples de ranking ;
  - cache les features visuelles par split dans `train|valid|test_item_visual_features.npz` ;
  - extrait maintenant les embeddings MobileNetV2 par batches avec `--embedding-batch-size` au lieu d'un `predict` image par image, pour mieux occuper le GPU Colab ;
  - conserve `model_status: experimental_only` par defaut.
- Diagnostic GPU renforce :
  - `configure_tensorflow_runtime` execute un petit `tf.matmul` sur `/GPU:0` quand un GPU est detecte ;
  - `metadata.json` expose `gpu_smoke_test_device` pour verifier que TensorFlow place bien une operation sur GPU.
- Ajout de `src/models/load_outfit_model.py` :
  - charge uniquement `models/outfit_active/` ;
  - refuse tout modele non promu, non V2 ou sans features image/couleur.
- Ajout de `src/services/outfit_v2_service.py` :
  - `recommend_associations_from_image` pour une image ;
  - `evaluate_outfit_images` pour plusieurs images ;
  - fallback cooccurrence/rules si V2 absent ou non promu.
- Mise a jour de `app/streamlit_app.py` :
  - onglet `Associer une piece` ;
  - onglet `Evaluer une tenue` ;
  - affichage scores ML/couleur/cooccurrence quand disponibles.
- Ajout de `src/analysis/analyze_outfit_v2_results.py` pour comparer V0, V1 et V2.
- Ajout de dependances Colab dans `requirements.txt` : `datasets`, `pyarrow`.
- Tests ajoutes :
  - features couleur V2 ;
  - interdiction des features ID ;
  - promotion metadata V2 ;
  - fallback mono-image sans modele promu ;
  - fallback multi-image sans modele promu.
- Validation intermediaire :
  - `python -m py_compile app\streamlit_app.py src\services\outfit_v2_service.py src\training\train_outfit_model_v2.py tests\test_services.py` OK ;
  - `.venv311\Scripts\python.exe -m pytest tests\test_services.py -k outfit_v2 --basetemp=.tmp_pytest -p no:cacheprovider` OK, 5 tests passes.
- Validation finale locale :
  - `python -m compileall app src tests` OK ;
  - `.venv311\Scripts\python.exe -m pytest --basetemp=.tmp_pytest -p no:cacheprovider` OK, 54 tests passes.
