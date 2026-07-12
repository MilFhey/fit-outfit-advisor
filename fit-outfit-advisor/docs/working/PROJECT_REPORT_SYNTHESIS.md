# Synthese projet - Fit & Outfit Advisor

## 1. Objectif du projet

Fit & Outfit Advisor est un prototype Streamlit/TensorFlow dont l'objectif est d'aider un utilisateur a obtenir :

- une reconnaissance du type de produit a partir d'une image ;
- une estimation prudente du fit/taille quand les donnees le permettent ;
- des recommandations de types de produits complementaires pour construire une tenue ;
- une interface de demonstration lisible, avec fallbacks lorsque les modeles ne sont pas suffisamment fiables.

Le projet a volontairement privilegie une logique fail-closed : aucun modele n'est utilise pour donner une recommandation ferme tant qu'il n'est pas explicitement promu, documente et charge depuis un dossier actif.

## 2. Architecture retenue

L'application est separee en couches :

- `app/streamlit_app.py` : interface Streamlit, sans logique metier lourde.
- `src/services/` : services applicatifs image, fit, outfit et conseil final.
- `src/mappings/` : mappings categorie/couleur/Fashion/Polyvore.
- `src/preprocessing/` : preparation des donnees tabulaires ou des paires outfit.
- `src/training/` : scripts d'entrainement experimentaux.
- `src/analysis/` : analyses post-entrainement, notamment abstention.
- `models/` : artefacts locaux hors Git ou placeholders.
- `reports/` : rapports JSON versionnables et decisions d'audit.
- `docs/working/` : documentation de travail, plans et conclusions.

Principe important : un dossier `models/*_v1` ou `models/*_v3` est experimental. Un modele devient actif uniquement s'il est copie volontairement dans `models/*_active/` avec des metadata promues.

## 3. Politique de promotion des modeles

Les services sont fail-closed :

- metadata absente -> fallback ;
- champ absent -> fallback ;
- `model_status` different de `"promoted"` -> fallback ;
- `promotable_to_streamlit` different de `true` -> fallback ;
- confiance trop faible -> sortie `unknown` ou `uncertain`.

Dossiers actifs :

- Fit : `models/fit_active/`.
- Image : `models/fashion_active/`.
- Outfit futur : `models/outfit_active/`.

Dossiers experimentaux :

- `models/fit_v2/`, `models/fit_v3/`, `models/fit_v3_all/`, `models/fit_v3_explicit/`.
- `models/fashion_v1/`.
- `models/outfit_v1/`.

## 4. Module ModCloth - recommandation de taille / fit

### 4.1 Objectif

Predire la classe `fit` parmi :

- `fit` ;
- `small` ;
- `large`.

But initial : determiner si un modele peut recommander prudemment une taille plus petite ou plus grande.

### 4.2 Donnees et variables

Dataset ModCloth :

- 82 790 lignes observees dans l'audit initial ;
- colonnes principales : `size`, `category`, `height`, `hips`, `bra size`, `cup size`, `fit` ;
- classes desequilibrees : environ 68,6 % `fit`, 15,7 % `large`, 15,7 % `small`.

Variables conservees uniquement si elles peuvent raisonnablement etre demandees a l'utilisateur avant achat :

- taille de vetement demandee ;
- categorie produit ;
- hauteur ;
- hanches ;
- taille de soutien-gorge ;
- bonnet.

Les variables type review text ou informations post-achat ont ete ecartees pour eviter la fuite de donnees.

### 4.3 Resultats V2

V2 ameliorait le baseline majoritaire :

- macro F1 test : `0.357` vs baseline `0.271` ;
- balanced accuracy test : `0.434` vs baseline `0.333` ;
- recall `small` : `0.502` ;
- recall `large` : `0.457`.

Mais V2 restait insuffisant pour une recommandation utilisateur :

- accuracy : `0.385` ;
- precision `small` : `0.209` ;
- precision `large` : `0.228` ;
- recall `fit` : `0.341`.

Decision : V2 est academique, non promouvable Streamlit.

### 4.4 Resultats V3

V3 a compare :

- baseline majoritaire ;
- regression logistique ;
- MLP TensorFlow ;
- MLP TensorFlow pondere avec `class_weight`.

Le meilleur run retenu pour analyse etait `fit_v3_all`.

Resultats principaux documentes :

- accuracy test : `0.4509` ;
- balanced accuracy test : `0.4338` ;
- macro F1 test : `0.3884` ;
- precision `large` environ `0.2340` ;
- precision `small` environ `0.2255`.

Le modele bat le baseline sur macro F1 et balanced accuracy, mais les precisions `small`/`large` restent trop faibles pour conseiller une taille.

### 4.5 Analyse d'abstention ModCloth

Une analyse de seuils de confiance a ete conduite sans reentrainement :

- seuils testes : `0.35` a `0.90` ;
- selection uniquement sur validation ;
- test evalue apres selection.

Critere souhaite :

- coverage validation >= 25 % ;
- precision `small` >= 0.40 ;
- precision `large` >= 0.40.

Resultat :

- aucun seuil acceptable ne respecte les contraintes ;
- decision : pas de recommandation ferme `small`/`large` dans Streamlit.

Conclusion ModCloth :

- resultat utile academiquement ;
- modele non promu ;
- service fit reste en fallback / uncertain.

## 5. Module Fashion Product Images - reconnaissance image

### 5.1 Objectif

Predire un `product_type_v0` depuis une image produit.

Taxonomie retenue :

- sortie modele : `product_type_v0` ;
- mapping metier derive : `canonical_category` ;
- role outfit derive : `outfit_role`.

Exemple :

```text
CNN -> shirt
canonical_category -> top
outfit_role -> top
```

### 5.2 Dataset

Dataset : Fashion Product Images Small.

Fichier principal :

- `styles.csv`

Images :

- dossier `images/`

La selection des classes V1 a ete faite apres audit reel des images lisibles et des effectifs.

### 5.3 Architecture testee

Deux approches prevues :

- CNN simple ;
- MobileNetV2 avec transfer learning.

MobileNetV2 a ete retenu comme meilleur candidat.

Important :

- CNN simple : normalisation `/255` uniquement ;
- MobileNetV2 : `preprocess_input` MobileNetV2 uniquement ;
- ne jamais appliquer les deux normalisations ensemble.

### 5.4 Resultats Fashion V1.1

Resultats experimentaux :

- architecture : MobileNetV2 ;
- accuracy test : `0.8748` ;
- balanced accuracy test : `0.8483` ;
- macro F1 test : `0.8526` ;
- weighted F1 test : `0.8725`.

La classe `flats` a ete retiree comme sortie visible et mappee vers `dress_shoes`, afin de stabiliser la taxonomie.

### 5.5 Abstention Fashion V1.1

Seuil de confiance retenu :

- `0.90`.

Resultats au seuil sur test :

- coverage : `0.7083` ;
- unknown rate : `0.2917` ;
- accuracy sur predictions non-unknown : `0.9695` ;
- macro F1 sur predictions non-unknown : `0.9425`.

Decision :

- Fashion V1.1 est promu localement dans `models/fashion_active/` ;
- le service image doit retourner `unknown` si la confiance est inferieure a `0.90` ;
- metadata active : `model_status: "promoted"` et `promotable_to_streamlit: true`.

## 6. Module Polyvore / Outfit Compatibility V0

### 6.1 Objectif

Le module outfit ne doit pas generer une tenue complete.

Il doit recommander des `product_type_v0` complementaires ordonnes.

Contrat V0 :

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

### 6.2 Alignement taxonomie

Polyvore ne doit pas creer une taxonomie independante.

Tout label Polyvore doit passer par :

```text
polyvore_label -> product_type_v0 -> canonical_category -> outfit_role
```

Roles autorises :

- `top` ;
- `bottom` ;
- `dress` ;
- `shoes` ;
- `outerwear` ;
- `bag` ;
- `accessory`.

### 6.3 Audit Hugging Face

Source principale :

- `mvasil/polyvore-outfits`.

Secret Colab :

- `HUGGIN_KEY`.

Premier constat :

- le loader `datasets` expose surtout `item_id` + `image` ;
- ce loader seul ne suffit pas pour la baseline cooccurrence.
- comptages loader observes : `disjoint` 71 967 / 14 657 / 70 035 lignes pour train/validation/test, `nondisjoint` 204 679 / 25 132 / 47 854 lignes pour train/validation/test.

Correction :

- inspection de la liste des fichiers Hugging Face ;
- telechargement des fichiers raw metadata ;
- cache Drive pour eviter de retelecharger.

### 6.4 Resultats raw metadata

Rapport officiel :

- `reports/polyvore_v0_dataset_audit.json`.

Resultats :

- `raw_metadata_ready_for_schema_audit: true` ;
- 29 fichiers raw metadata ;
- `disjoint/train.json` : 16 995 outfits ;
- `disjoint/valid.json` : 3 000 outfits ;
- `disjoint/test.json` : 15 145 outfits ;
- `nondisjoint/train.json` : 53 306 outfits ;
- `nondisjoint/valid.json` : 5 000 outfits ;
- `nondisjoint/test.json` : 10 000 outfits ;
- `polyvore_item_metadata.json` : 251 008 items ;
- `polyvore_outfit_titles.json` : 68 306 outfits.

Champs detectes dans `polyvore_item_metadata.json` :

- `category_id` ;
- `catgeories` ;
- `semantic_category` ;
- `title` ;
- `description` ;
- `url_name`.

Decision :

- dataset exploitable pour schema/mapping ;
- pas d'entrainement encore ;
- prochaine etape : mapping Polyvore vers Fashion V1 puis baseline cooccurrence.

### 6.5 Audit schema/mapping Polyvore

Un script reproductible a ete ajoute pour transformer l'audit raw en audit de schema/mapping :

- script : `src/analysis/analyze_polyvore_v0_schema_mapping.py` ;
- rapport cible : `reports/polyvore_v0_schema_mapping_audit.json` ;
- entree attendue : dossier raw HF contenant `polyvore_item_metadata.json`, `categories.csv` et les splits `disjoint`/`nondisjoint`.

Le script :

- inspecte la structure exacte des `items` dans les splits ;
- relie les items aux metadata ;
- extrait les distributions des champs `semantic_category`, `category_id`, `catgeories`, `title` et `url_name` ;
- propose un mapping vers la taxonomie Fashion V1.1 ;
- documente les exclusions ;
- force `training_executed: false` et `streamlit_integration_executed: false`.

Etat local observe : les fichiers raw HF ne sont pas presents dans `data/raw/`, donc le rapport local est marque `raw_files_missing_requires_colab_or_drive_raw_root`. Les distributions reelles doivent etre regenerees dans Colab ou dans un environnement disposant du cache Drive raw.

Le fichier fourni `models/polyvore/polyvore_v0_dataset_audit (2).json` a ete compare au rapport `reports/polyvore_v0_dataset_audit.json` et ne presente aucune difference. Le rapport schema/mapping reprend donc ce rapport versionne comme entree officielle et y ajoute le resume loader/raw.

Resultat Colab apres execution sur raw HF :

- decision : `schema_mapping_ready_for_manual_review` ;
- raw root : `/content/fit-outfit-runtime/polyvore/raw_hf_files` ;
- items lies aux metadata : 540 539 ;
- mappings proposes : 50 ;
- labels exclus documentes : 50.

Conclusion : Polyvore est exploitable pour une baseline cooccurrence, mais uniquement apres revue manuelle et promotion explicite du mapping. Les labels generiques ou hors taxonomie Fashion V1.1 restent exclus.

Revue manuelle effectuee :

- `config/outfit_v1_config.json` passe en `validated_for_baseline_v0` ;
- 28 labels Polyvore fiables sont promus vers Fashion V1.1 ;
- les labels generiques et hors taxonomie restent exclus ;
- la prochaine etape devient la baseline cooccurrence, sans TensorFlow.

## 7. Decisions importantes

| Sujet | Decision | Justification |
|---|---|---|
| ModCloth V2/V3 | Non promu | Precision `small`/`large` trop faible |
| Fit service | Fallback/uncertain | Pas de seuil d'abstention acceptable |
| Fashion V1.1 | Promu localement | Bonnes metriques + abstention stable |
| Image service | Actif avec seuil 0.90 | `unknown` sous le seuil |
| Polyvore | Pas d'entrainement encore | Schema/mapping a valider d'abord |
| Outfit V0 | Baseline cooccurrence d'abord | Plus interpretable et adaptee au MVP |
| Artefacts | Fail-closed | Eviter toute recommandation non fiable |

## 8. Etat actuel du MVP

Fonctionnel :

- interface Streamlit ;
- fallback image ;
- modele image Fashion actif localement si artefacts presents ;
- fallback fit prudent ;
- service outfit rule-based/fallback ;
- conseil final structure ;
- tests unitaires MVP.

Non finalise :

- vrai modele fit promu ;
- baseline cooccurrence Polyvore ;
- modele TensorFlow outfit ;
- integration avancee de recommandations outfit dans Streamlit.

## 9. Limites actuelles

### Fit

- precision trop faible sur les classes minoritaires ;
- dataset subjectif, bruit important ;
- variables utilisateur incompletes ou manquantes ;
- pas de recommandation de taille fiable.

### Image

- modele performant mais encore dependant de la qualite des images produit ;
- abstention necessaire pour les classes difficiles ;
- classes hors taxonomie V1.1 doivent retourner `unknown`.

### Outfit

- mapping Polyvore -> Fashion V1 pas encore construit ;
- cooccurrences a calculer proprement ;
- fuite de donnees a controler entre splits ;
- item_id interdit comme feature directe de modele.

## 10. Prochaines etapes recommandees

### Priorite immediate

Construire la baseline cooccurrence Polyvore :

1. Charger les splits raw `disjoint`/`nondisjoint`.
2. Relier les items a `polyvore_item_metadata.json`.
3. Appliquer `config/outfit_v1_config.json`.
4. Extraire les paires positives d'items d'un meme outfit.
5. Agreger au niveau `product_type_v0` et `outfit_role`.
6. Calculer des scores de cooccurrence normalises.
7. Verifier l'absence de fuite train/validation/test.

Implementation amorcee :

- script : `src/analysis/build_polyvore_v0_cooccurrence_baseline.py` ;
- rapport : `reports/polyvore_v0_cooccurrence_baseline.json` ;
- statut local : fail-closed, raw HF absents du workspace ;
- execution attendue : Colab, cellule `Baseline cooccurrence Polyvore V0`.

La baseline reste interpretable et non TensorFlow. Elle produit des recommandations par `product_type_v0` et un score brut de cooccurrence pour preparer l'integration experimentale future.

### Ensuite

Construire la baseline cooccurrence :

1. Extraire les paires positives d'items d'un meme outfit.
2. Agreger au niveau `product_type_v0`.
3. Calculer scores de cooccurrence normalises.
4. Evaluer ranking : Precision@K, Recall@K, MRR ou NDCG@K.
5. Verifier absence de fuite entre splits.
6. Integrer en fallback/rule-based si robuste.

### Plus tard seulement

Entrainer un modele TensorFlow outfit si :

- la baseline est propre ;
- le mapping est valide ;
- les erreurs principales sont documentees ;
- le modele bat clairement la baseline.

## 11. Commandes de validation utilisees

Commandes regulierement executees :

```bash
python -m compileall app src tests
pytest --basetemp=.tmp_pytest -p no:cacheprovider
```

Resultat courant observe :

- 43 tests passent.

## 12. Fichiers de reference pour le rapport

- `docs/working/CODEX_AUDIT.md`
- `docs/working/MODCLOTH_V3_EXPERIMENT_PLAN.md`
- `docs/working/FASHION_CNN_V0_PLAN.md`
- `docs/working/OUTFIT_COMPATIBILITY_V0_PLAN.md`
- `reports/modcloth_v3_abstention_all.json`
- `reports/modcloth_v3_abstention_explicit.json`
- `reports/polyvore_v0_dataset_audit.json`
- `models/fashion_v1/fashion_v1_abstention.json`
- `models/README.md`
- `README.md`

## 13. Conclusion generale

Le projet a evolue d'un prototype simule vers un MVP structure :

- ModCloth a ete analyse rigoureusement mais non promu, car la fiabilite utilisateur n'est pas suffisante.
- Fashion V1.1 est le premier module vraiment promu, grace a de bonnes performances et une abstention explicite.
- Polyvore devient exploitable apres recuperation des fichiers raw metadata, mais reste au stade audit/mapping avant baseline.

La decision la plus importante du projet est de ne pas forcer les modeles faibles dans l'application. Le systeme privilegie les resultats fiables, documentes et abstention-aware.
