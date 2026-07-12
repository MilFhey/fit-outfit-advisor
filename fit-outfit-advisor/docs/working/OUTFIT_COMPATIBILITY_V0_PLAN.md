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
- Le rapport local est fail-closed (`raw_files_missing_requires_colab_or_drive_raw_root`) car les raw HF ne sont pas dans le workspace.
- Le notebook `03_polyvore_exploration_colab.ipynb` genere maintenant le vrai rapport complet dans Drive via la cellule `Baseline cooccurrence Polyvore V0`.
- La baseline primaire est calculee uniquement depuis `disjoint_train`. Les aggregats complets par config restent des diagnostics, pas une source de scoring primaire.
- Les recouvrements entre `disjoint` et `nondisjoint` sont conserves comme diagnostic seulement, car ce sont deux configurations alternatives du dataset.
- Si des paires exactes fuient entre `train` et validation/test, la baseline reste construite mais l'evaluation doit filtrer ces paires ou rester bloquee.
- Prochaine decision apres execution Colab : relire les recommandations `primary_baseline` par `product_type_v0`, regarder `baseline_decision`, puis integrer seulement si l'evaluation est propre ou explicitement filtree.

## Promotion future
- `models/outfit_v1/` reste experimental.
- `models/outfit_active/` est le seul emplacement actif.
- Fail-closed obligatoire : metadata absente, invalide, non promue ou artefact absent -> fallback rule-based.
- Promotion seulement si le modele bat clairement la baseline cooccurrence et si les erreurs principales sont documentees.
