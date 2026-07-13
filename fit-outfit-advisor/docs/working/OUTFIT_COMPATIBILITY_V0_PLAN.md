# OUTFIT_COMPATIBILITY_V0_PLAN - Polyvore / Outfit Compatibility

## Statut
- Prochain module apres promotion controlee Fashion V1.1.
- Objectif V0 : recommander des types de produits complementaires, pas generer une tenue complete.
- Priorite stricte : audit dataset, baseline cooccurrence, puis entrainement TensorFlow seulement si le dataset est exploitable.
- Aucun entrainement ne doit etre lance avant validation du rapport `reports/polyvore_v0_dataset_audit.json`.

## Alignement taxonomie Fashion V1
- Polyvore ne cree pas de taxonomie independante.
- Concepts obligatoires :
  - `product_type_v0` ;
  - `canonical_category` ;
  - `outfit_role`.
- Tout label Polyvore doit passer par un mapping explicite :
  - `polyvore_label -> product_type_v0` ;
  - `product_type_v0 -> canonical_category` ;
  - `canonical_category -> outfit_role`.
- Roles autorises V0 : `top`, `bottom`, `dress`, `shoes`, `outerwear`, `bag`, `accessory`.
- Le mapping officiel brouillon est `config/outfit_v1_config.json`.

## Livrable audit Colab
- Notebook officiel : `notebooks/03_polyvore_exploration_colab.ipynb`.
- Le notebook doit :
  - monter Google Drive ;
  - cloner ou mettre a jour le repo ;
  - installer les dependances ;
  - creer `/content/fit-outfit-runtime/polyvore/...` ;
  - charger la source principale Hugging Face `mvasil/polyvore-outfits` avec le secret Colab `HUGGIN_KEY` ;
  - privilegier les configurations/splits `disjoint` et `nondisjoint` si disponibles ;
  - sauvegarder aussi le dataset dans Google Drive sous `MyDrive/fit-outfit-advisor/datasets/mvasil_polyvore_outfits` ;
  - verifier d'abord cette copie Drive pour eviter de retelecharger le dataset ;
  - afficher les fichiers Hugging Face, les `features` chargees et les cles de premiere ligne ;
  - detecter les fichiers outfits/items/images/metadonnees ;
  - inspecter shapes, colonnes, valeurs manquantes, categories, outfits et items par outfit ;
  - generer `reports/polyvore_v0_dataset_audit.json` ;
  - s'arreter explicitement avant entrainement.

## Contrat de sortie V0
```python
{
    "input_product_type": "shirt",
    "recommended_product_types": ["jeans", "trousers", "casual_shoes"],
    "compatible_roles": ["bottom", "shoes"],
    "raw_compatibility_score": 0.0,
    "mode": "rule_based | model | fallback",
    "model_status": "experimental_only | promoted | fallback",
    "reason": "..."
}
```

`outfit_service` conserve aussi les cles historiques `compatible_items`, `compatible_colors` et `compatibility_score`.

## Baseline cooccurrence
- Extraire les paires positives depuis les items d'un meme outfit.
- Agreger les paires au niveau `product_type_v0` et `outfit_role`.
- Produire des scores normalises pour classer des types complementaires.
- La baseline doit devenir le premier candidat exploitable du service avant tout modele TensorFlow.

## Resultat audit `mvasil/polyvore-outfits` - 2026-07-09
- Rapport local : `reports/polyvore_v0_dataset_audit.json`.
- Configurations disponibles : `disjoint` et `nondisjoint`.
- Colonnes observees dans les splits Hugging Face : `item_id`, `image`.
- Comptages loader `datasets` confirmes par Colab :
  - `disjoint/train` : 71 967 lignes ;
  - `disjoint/validation` : 14 657 lignes ;
  - `disjoint/test` : 70 035 lignes ;
  - `nondisjoint/train` : 204 679 lignes ;
  - `nondisjoint/validation` : 25 132 lignes ;
  - `nondisjoint/test` : 47 854 lignes.
- Le loader `datasets` seul ne suffit pas pour Outfit V0 : il n'expose pas directement `outfit_id`/`set_id`, labels de categorie/type, ni composition des outfits.
- La liste des fichiers HF contient toutefois les metadata brutes attendues : `categories.csv`, `polyvore_item_metadata.json`, `polyvore_outfit_titles.json`, `disjoint/*.json`, `nondisjoint/*.json`, `compatibility_*.txt`.
- Decision revisee : source Hugging Face probablement exploitable via fichiers raw ; il faut inspecter leur schema avant baseline cooccurrence.
- Prochaine action : telecharger/cache Drive les fichiers raw HF, verifier leurs cles et construire le mapping Polyvore -> Fashion V1.

## Resultat audit raw metadata - 2026-07-11
- Rapport officiel : `reports/polyvore_v0_dataset_audit.json`.
- `raw_metadata_ready_for_schema_audit: true`.
- Fichiers raw telecharges et caches Drive : 29.
- Outfits disponibles :
  - `disjoint/train.json` : 16 995 outfits ;
  - `disjoint/valid.json` : 3 000 outfits ;
  - `disjoint/test.json` : 15 145 outfits ;
  - `nondisjoint/train.json` : 53 306 outfits ;
  - `nondisjoint/valid.json` : 5 000 outfits ;
  - `nondisjoint/test.json` : 10 000 outfits.
- `polyvore_item_metadata.json` : 251 008 items avec `category_id`, `catgeories`, `semantic_category`, `title`, `description`, `url_name`.
- `polyvore_outfit_titles.json` : 68 306 outfits.
- Decision : audit dataset valide pour passer a l'etape schema/mapping avant baseline.
- Prochaine action : analyser `items` dans les splits, extraire les paires positives, mapper `semantic_category`/`category_id` vers Fashion V1 puis generer la baseline cooccurrence.

## Audit schema/mapping local - 2026-07-11
- Nouveau script : `src/analysis/analyze_polyvore_v0_schema_mapping.py`.
- Nouveau rapport : `reports/polyvore_v0_schema_mapping_audit.json`.
- Notebook `notebooks/03_polyvore_exploration_colab.ipynb` mis a jour : apres le rapport dataset, il genere aussi le rapport schema/mapping, l'affiche en apercu et le copie dans Drive.
- Le fichier `models/polyvore/polyvore_v0_dataset_audit (2).json` a ete verifie comme identique a `reports/polyvore_v0_dataset_audit.json` ; le rapport versionne reste la reference.
- Le script lit les fichiers raw HF attendus :
  - `polyvore_item_metadata.json` ;
  - `categories.csv` ;
  - `disjoint/{train,valid,test}.json` ;
  - `nondisjoint/{train,valid,test}.json`.
- Il inspecte la structure des splits, relie les `item_id` candidats a `polyvore_item_metadata.json`, extrait les distributions `semantic_category`, `category_id`, `catgeories`, `title` et `url_name`, puis propose un mapping vers `product_type_v0 -> canonical_category -> outfit_role`.
- Etat local actuel : les fichiers raw HF ne sont pas presents dans `data/raw/`; le rapport local est donc volontairement marque `raw_files_missing_requires_colab_or_drive_raw_root`.
- Le rapport schema/mapping reprend maintenant le resume de l'audit Colab : `loader_only_cooccurrence_possible: false`, `cooccurrence_baseline_possible: true`, split rows du loader et fichiers raw cles.
- Le rapport local contient quand meme la politique de mapping/exclusion reproductible, alignee avec Fashion V1.1 :
  - roles retenus : `top`, `bottom`, `dress`, `shoes`, `outerwear`, `bag`, `accessory` ;
  - exclusions explicites : beauty/cosmetics, fragrance, home/decor, electronics, headwear non retenu, hosiery, underwear, swimwear, boots, skirts et scarves sans `product_type_v0` fidele.
- Commande a executer dans Colab ou dans un environnement ou le cache Drive est monte :

```bash
python -m src.analysis.analyze_polyvore_v0_schema_mapping \
  --raw-root /content/drive/MyDrive/fit-outfit-advisor/datasets/mvasil_polyvore_outfits_raw_files \
  --output reports/polyvore_v0_schema_mapping_audit.json
```

- Decision : ne pas remplir `config/outfit_v1_config.json` ni lancer la baseline cooccurrence tant que les distributions reelles et les labels exclus n'ont pas ete calcules depuis les raw.

## Resultat schema/mapping Colab - 2026-07-11
- Le notebook a genere `reports/polyvore_v0_schema_mapping_audit.json` depuis `/content/fit-outfit-runtime/polyvore/raw_hf_files`.
- Decision : `schema_mapping_ready_for_manual_review`.
- Items relies aux metadata : 540 539.
- Mappings proposes : 50.
- Labels exclus documentes : 50.
- Interpretation :
  - la source raw est exploitable pour Outfit V0 ;
  - les labels forts et coherents avec Fashion V1.1 sont nombreux : `shoes`, `bags`, `jewellery`, `tops`, `sunglasses`, `sandals`, `jeans`, `pumps`, `sneakers`, `pants`, `sweaters`, `blouses`, `shorts`, `t shirts`, `flats`, `jackets` ;
  - les labels larges comme `women s fashion`, `clothing`, `all body`, `accessories` restent a exclure car ils ne correspondent pas a un `product_type_v0` precis ;
  - les labels hors taxonomie Fashion V1.1 (`skirts`, `boots`, `hats`, `scarves`, swimwear, leggings, underwear) doivent rester exclus pour eviter une taxonomie outfit parallele.
- Corrections appliquees aux regles d'audit apres lecture du resultat :
  - `outerwear`, `hoodies` et `vests` sont maintenant mappes vers `outerwear` ;
  - `capri cropped pants` n'est plus exclu par faux positif sur `cap` et peut etre mappe vers `trousers` ;
  - `converse` / `chuck taylor` sont mappables vers `sports_shoes`.
- Le notebook contient maintenant un sanity check avant generation du rapport schema/mapping. Il doit afficher ces mappings comme `mapped` avant d'ecrire le rapport :
  - `outerwear -> outerwear` ;
  - `hoodies -> outerwear` ;
  - `vests -> outerwear` ;
  - `capri cropped pants -> trousers` ;
  - `converse chuck taylor all star -> sports_shoes`.
- Decision apres revue manuelle : promotion d'un noyau conservateur dans `config/outfit_v1_config.json`.
- Config Outfit V1 :
  - `status`: `validated_for_baseline_v0` ;
  - `source_label_column`: `semantic_category|category_id_name|catgeories` ;
  - 28 labels Polyvore promus vers Fashion V1.1.
- Labels promus principaux : `shoes`, `bags`, `jewellery`, `tops`, `outerwear`, `sunglasses`, `sandals`, `jeans`, `pumps`, `sneakers`, `pants`, `sweaters`, `blouses`, `shorts`, `t shirts`, `flats`, `sweatshirts hoodies`, `converse sneakers`, `capri cropped pants`, `hoodies`, `vests`.
- Labels non promus volontairement : labels generiques (`women s fashion`, `clothing`, `all body`, `accessories`) et labels hors taxonomie Fashion V1.1 (`skirts`, `boots`, `hats`, `scarves`, swimwear, leggings, underwear).
- Prochaine etape : construire la baseline cooccurrence sur les raw HF avec ce mapping valide.

## Negatifs difficiles futurs
- Generer des negatifs avec roles compatibles.
- Remplacer un item par un autre item du meme role, et de meme famille quand possible.
- Exclure toute paire qui apparait comme positive dans un autre outfit.
- Documenter le ratio positif/negatif et utiliser un seed fixe.
- Exemples souhaites : `shirt + autre jeans`.
- Eviter les negatifs triviaux comme `shirt + perfume` ou `tshirt + lipstick`.

## Split et fuite
- Split groupe par `outfit_id`.
- `item_id` et `outfit_id` sont interdits comme features directes.
- Verifier qu'aucune paire positive exacte ne fuite entre train/validation/test.
- Documenter les limites si un meme item apparait dans plusieurs outfits.

## Metriques futures
- Classification :
  - balanced accuracy ;
  - ROC AUC ;
  - macro F1.
- Ranking :
  - Precision@K ;
  - Recall@K ;
  - NDCG@K ou MRR.

## Baseline cooccurrence V0
- Nouveau script : `src/analysis/build_polyvore_v0_cooccurrence_baseline.py`.
- Nouveau rapport : `reports/polyvore_v0_cooccurrence_baseline.json`.
- Le script :
  - charge les splits raw Polyvore ;
  - relie les items a `polyvore_item_metadata.json` ;
  - applique `config/outfit_v1_config.json` en `validated_for_baseline_v0` ;
  - extrait les paires positives compatibles dans un meme outfit ;
  - agrege les cooccurrences dirigees par `product_type_v0` ;
  - calcule un `raw_compatibility_score` simple : count candidat / total cooccurrences de l'input ;
  - verifie les recouvrements de paires positives exactes entre splits au sein de chaque config (`disjoint` puis `nondisjoint`).
- Aucun negatif, aucun modele TensorFlow et aucune integration Streamlit ne sont generes a cette etape.
- Le rapport versionne `reports/polyvore_v0_cooccurrence_baseline.json` reprend le dernier resume Colab transmis. Le JSON complet Colab n'est pas present localement, donc le rapport indique `report_completeness: summary_only_full_colab_json_not_available_locally`.
- Le notebook `03_polyvore_exploration_colab.ipynb` genere le vrai rapport complet dans Drive via la cellule `Baseline cooccurrence Polyvore V0`.
- La baseline primaire est calculee uniquement depuis `disjoint_train`. Les aggregats complets par config restent des diagnostics, pas une source de scoring primaire.
- Les recouvrements entre `disjoint` et `nondisjoint` sont conserves comme diagnostic seulement, car ce sont deux configurations alternatives du dataset.
- Si des paires exactes fuient entre `train` et validation/test, la baseline reste construite mais l'evaluation doit filtrer ces paires ou rester bloquee.
- Dernier resultat Colab transmis :
  - `baseline_ready: true` ;
  - `tensorflow_used: false` ;
  - config primaire : `disjoint` ;
  - split d'entrainement primaire : `disjoint_train` ;
  - paires dirigees split primaire : `92390` ;
  - paires `product_type_v0` uniques split primaire : `26` ;
  - `baseline_decision: train_only_baseline_ready_with_leakage_filtered_evaluation`.
- Apercu des recommandations primaires :
  - `top` -> `casual_shoes`, `bag`, `outerwear`, `jeans`, `trousers` ;
  - `jeans` -> `top`, `casual_shoes`, `outerwear` ;
  - `outerwear` -> `top`, `jeans`, `trousers`, `shorts` ;
  - `casual_shoes` -> `bag`, `top`, `jeans`, `trousers`, `shorts`.
- Fuite positive exacte intra-config :
  - `disjoint train__valid`: `28` ;
  - `disjoint train__test`: `0` ;
  - `nondisjoint train__valid`: `61` ;
  - `nondisjoint train__test`: `160` ;
  - `nondisjoint valid__test`: `15`.
- Mise a jour implementation locale :
  - `src/analysis/build_polyvore_v0_cooccurrence_baseline.py` calcule maintenant `leakage_filtered_evaluation` ;
  - les paires positives exactes deja vues dans `train` sont retirees de validation/test avant calcul des metriques ;
  - metriques produites par split : `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr`, compte brut, compte filtre et compte evaluable ;
  - `baseline_decision` passe a `train_only_baseline_ready_with_leakage_filtered_evaluation` si au moins un split d'evaluation reste evaluable apres filtrage.
- Dernier rerun Colab avec le nouvel affichage :
  - `leakage_filtered_evaluation_ready: true` ;
  - valid : `evaluable_directed_pair_count=16492`, `mrr=0.711036`, `recall_at_k_3=0.917596`, `ndcg_at_k_3=0.751335` ;
  - test : `evaluable_directed_pair_count=80212`, `mrr=0.709846`, `recall_at_k_3=0.929375`, `ndcg_at_k_3=0.755142`.
- Decision : les metriques filtrees sont defendables et stables entre valid/test pour une integration experimentale fail-closed.
- Integration locale : `src/services/outfit_service.py` peut maintenant utiliser `reports/polyvore_v0_cooccurrence_baseline.json` en mode `cooccurrence_baseline` seulement si le rapport est explicitement pret, sans TensorFlow, avec fallback rule-based sinon.

## Promotion future
- `models/outfit_v1/` reste experimental.
- `models/outfit_active/` est le seul emplacement actif.
- Fail-closed obligatoire : metadata absente, invalide, non promue ou artefact absent -> fallback rule-based.
- Promotion seulement si le modele bat clairement la baseline cooccurrence et si les erreurs principales sont documentees.

## Outfit Compatibility V1 TensorFlow experimental - 2026-07-12
- Objectif : ajouter une vraie modelisation TensorFlow pour l'enonce du cours, sans remplacer la baseline MVP tant qu'une revue de promotion n'a pas ete faite.
- Nouveau script : `src/training/train_outfit_model_v1.py`.
- Tache : classification binaire `compatible` / `not_compatible` sur des paires de types produits Polyvore.
- Donnees :
  - positives : paires dirigees d'items compatibles issues d'un meme outfit, apres mapping Polyvore -> Fashion V1.1 ;
  - negatives : negatifs difficiles, en remplacant si possible le candidat par un item d'un autre outfit mais du meme role/famille ;
  - split primaire : `disjoint`, conserve par outfit/set.
- Garde-fous :
  - `item_id` et `outfit_id` interdits comme features directes ;
  - paires positives exactes vues en train filtrees de validation/test ;
  - seuil de decision choisi uniquement sur validation ;
  - comparaison systematique avec la baseline cooccurrence ;
  - artefacts ecrits dans `models/outfit_v1/` avec `model_status: experimental_only` et `promotable_to_streamlit: false`.
- Features autorisees : colonnes de `OUTFIT_PAIR_FEATURE_COLUMNS`, c'est-a-dire types produits, categories canoniques, roles outfit et indicateurs de relation entre roles/types.
- Sorties attendues :
  - `models/outfit_v1/outfit_model.keras` ;
  - `models/outfit_v1/outfit_preprocessor.joblib` ;
  - `models/outfit_v1/metadata.json` ;
  - `models/outfit_v1/metrics.json` ;
  - matrices de confusion et courbe d'entrainement.
- Commande Colab recommandee :

```bash
python -m src.training.train_outfit_model_v1 \
  --raw-root /content/fit-outfit-runtime/polyvore/raw_hf_files \
  --output-dir models/outfit_v1 \
  --epochs 25 \
  --batch-size 256 \
  --require-gpu
```

- Decision MVP : meme si le modele TensorFlow est entraine, l'application continue a utiliser la baseline cooccurrence fail-closed tant que `models/outfit_active/` n'est pas explicitement promu.
- Diagnostic GPU : le script ecrit `tensorflow_device_summary` dans `metadata.json` et `metrics.json`. Si `--require-gpu` est passe et que TensorFlow ne voit pas le T4, l'entrainement s'arrete avec une erreur explicite.

## Resultat Outfit V1 TensorFlow - 2026-07-12
- Artefacts analyses : `models/outfit_v1/`.
- Rapport d'analyse : `reports/outfit_v1_training_analysis.json`.
- Donnees :
  - train : 184 780 paires labellisees, equilibrees 92 390 / 92 390 ;
  - valid : 33 042 paires labellisees ;
  - test : 160 424 paires labellisees, equilibrees 80 212 / 80 212.
- Resultat test TensorFlow MLP :
  - accuracy : `0.5008` ;
  - balanced accuracy : `0.5008` ;
  - macro F1 : `0.3611` ;
  - ROC AUC : `0.5016` ;
  - recall `compatible` : `0.9684` ;
  - recall `not_compatible` : `0.0332` ;
  - MRR produit experimental : `0.3632` ;
  - recall@3 produit experimental : `0.3846`.
- Diagnostic :
  - loss proche de `0.693`, donc signal quasi aleatoire ;
  - AUC train/validation autour de `0.50` ;
  - le modele predit presque tout en `compatible` ;
  - le modele ne bat pas la baseline cooccurrence sur le critere de promotion.
- Cause probable :
  - les features visibles par le MLP sont seulement `product_type_v0`, `canonical_category` et `outfit_role` ;
  - les negatifs sont construits au niveau item, mais le modele ne voit ni `item_id`, ni image, ni couleur, ni texte, ni attribut de style ;
  - beaucoup de paires negatives peuvent donc etre indistinguables de paires positives au niveau feature.
- Decision :
  - ne pas promouvoir `models/outfit_v1/` ;
  - conserver l'experience dans le rapport comme tentative TensorFlow rigoureuse mais non concluante ;
  - garder le MVP sur baseline cooccurrence fail-closed + futures regles couleur/outfit.

## Outfit Compatibility V2 multimodal - sprint implementation 2026-07-12
- Objectif : mettre le machine learning au centre du module outfit en apprenant une compatibilite pairwise depuis image + couleur + taxonomie + score cooccurrence.
- Nouveau preprocessing : `src/preprocessing/outfit_v2_features.py`.
  - extrait une couleur dominante via `MiniBatchKMeans` ;
  - classe la famille couleur (`black`, `white`, `blue`, `red`, etc.) ;
  - calcule un score d'harmonie couleur code ;
  - extrait un embedding MobileNetV2 ImageNet 1280 dimensions ;
  - construit les features pairwise : embeddings input/candidat, difference absolue, similarite cosinus, distance L2, couleurs, harmonie et cooccurrence.
- Nouveau script TensorFlow : `src/training/train_outfit_model_v2.py`.
  - utilise les raw Polyvore pour les outfits et le loader Hugging Face pour `item_id + image` ;
  - conserve `disjoint` comme config primaire ;
  - construit des paires positives depuis les outfits et des negatifs difficiles via le builder V1 ;
  - interdit `item_id`, `outfit_id` et `set_id` comme features directes ;
  - choisit le seuil uniquement sur validation ;
  - compare V2 a la baseline cooccurrence ;
  - cache les embeddings/couleurs par split dans `*_item_visual_features.npz` pour eviter de recalculer MobileNetV2 a chaque run ;
  - extrait les embeddings MobileNetV2 en batch via `--embedding-batch-size` pour utiliser le GPU T4 pendant la phase image ;
  - ecrit les artefacts dans `models/outfit_v2/`.
- Artefacts attendus :
  - `outfit_model.keras` ;
  - `outfit_preprocessor.joblib` ;
  - `product_type_prototypes.json` ;
  - `metadata.json` ;
  - `metrics.json` ;
  - `training_history.png` ;
  - `confusion_matrix_raw.png` ;
  - `ranking_examples.png`.
- Nouveau script d'analyse : `src/analysis/analyze_outfit_v2_results.py`.
  - compare `cooccurrence_v0`, `outfit_v1_tensorflow_mlp` et `outfit_v2_multimodal` ;
  - genere `reports/outfit_v2_results_analysis.json` ;
  - recommande explicitement promotion ou maintien experimental.
- Nouveau service applicatif : `src/services/outfit_v2_service.py`.
  - mode `recommend_associations_from_image` : une image -> detection Fashion V1 -> couleur -> recommandations ;
  - mode `evaluate_outfit_images` : plusieurs images -> score global de tenue, scores par paire, roles manquants, suggestions ;
  - charge `models/outfit_active/` seulement si les metadata indiquent `version=outfit_v2`, `model_status=promoted`, `promotable_to_streamlit=true`, `uses_image_embeddings=true`, `uses_color_features=true`.
- Streamlit expose maintenant deux onglets :
  - `Associer une piece` ;
  - `Evaluer une tenue`.
- Critere de promotion V2 :
  - ROC AUC test >= `0.60` ;
  - Recall@3 test >= baseline cooccurrence ou gain clair sur MRR ;
  - pas de collapse de classe ;
  - erreurs principales documentees ;
  - promotion manuelle vers `models/outfit_active/` uniquement.
- Commande Colab recommandee :

```bash
python -m src.training.train_outfit_model_v2 \
  --raw-root /content/fit-outfit-runtime/polyvore/raw_hf_files \
  --output-dir models/outfit_v2 \
  --epochs 10 \
  --batch-size 128 \
  --embedding-batch-size 128 \
  --embedding-backend tf_data \
  --color-extraction-mode fast \
  --recompute-visual-cache \
  --require-gpu
```

- Diagnostic GPU :
  - `--require-gpu` arrete le script si TensorFlow ne voit aucun GPU ;
  - `metadata.json` contient `tensorflow_device_summary.gpu_available` et `gpu_smoke_test_device` ;
  - les logs affichent `Outfit V2 embedding device requested` et `embedding_devices` par split ;
  - TensorFlow est initialise des le debut de `train()` pour lancer un smoke test GPU avant les phases CPU ;
  - l'extraction visuelle lit Hugging Face en streaming et envoie les images a MobileNetV2 par batch sur `/GPU:0`, au lieu de charger toutes les images avant le premier calcul GPU ;
  - la colonne Hugging Face `image` est forcee en `decode=False` pour eviter le decodage CPU des lignes ignorees pendant le scan ;
  - les lignes Hugging Face sont selectionnees par index `item_id -> row_index` pour eviter une iteration Python complete du split ;
  - `--embedding-backend tf_data` decode/resize/preprocess les images via TensorFlow en batch au lieu d'une boucle PIL/Numpy image par image ;
  - `--embedding-batch-size 128` est recommande sur T4 ; le script reduit automatiquement les sous-batches MobileNetV2 si une erreur OOM GPU apparait ;
  - `--color-extraction-mode fast` est le mode recommande Colab pour eviter que le clustering couleur CPU masque l'utilisation GPU.

- Decision MVP : tant que V2 n'est pas promu, l'application utilise la baseline cooccurrence/rules mais l'interface et le service sont prets pour le modele actif.
