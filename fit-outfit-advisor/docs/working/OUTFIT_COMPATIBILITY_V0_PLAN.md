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
  - accepter une source Kaggle ou un zip Google Drive ;
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

## Promotion future
- `models/outfit_v1/` reste experimental.
- `models/outfit_active/` est le seul emplacement actif.
- Fail-closed obligatoire : metadata absente, invalide, non promue ou artefact absent -> fallback rule-based.
- Promotion seulement si le modele bat clairement la baseline cooccurrence et si les erreurs principales sont documentees.
