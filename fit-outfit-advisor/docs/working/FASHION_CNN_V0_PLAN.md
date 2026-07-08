# FASHION_CNN_V0_PLAN - Fashion Product Images Small

## Statut
- Nouvelle priorite apres cloture du module ModCloth comme experimentation academique.
- Dataset inspecte dans Colab ; configuration V1 validee pour entrainement.
- Aucun entrainement image local ne doit etre lance tant que le dataset image n'est pas present.
- Le notebook d'inspection officiel est `notebooks/02_train_fashion_model_colab.ipynb`.

## Dataset attendu
- Source Kaggle attendue : `paramaggarwal/fashion-product-images-small`.
- Structure attendue apres extraction :
  - `styles.csv` : metadonnees produit ;
  - `images/` : fichiers image produit ;
  - convention attendue : une ligne `styles.csv.id` correspond a `images/{id}.jpg`.
- Colonnes attendues a confirmer dans le notebook :
  - `id` ;
  - `gender` ;
  - `masterCategory` ;
  - `subCategory` ;
  - `articleType` ;
  - `baseColour` ;
  - `season` ;
  - `year` ;
  - `usage` ;
  - `productDisplayName`.

## Cible V0
- Cible apprise par le CNN : `product_type_v0`.
- Colonne source : `articleType`.
- Le modele doit predire un type produit visible et plus informatif qu'une categorie metier :
  - `tshirt` ;
  - `shirt` ;
  - `top` ;
  - `jeans` ;
  - `trousers` ;
  - `shorts` ;
  - `dress` ;
  - `outerwear` ;
  - `casual_shoes` ;
  - `sports_shoes` ;
  - `dress_shoes` ;
  - `sandals` ;
  - `flip_flops` ;
  - `heels` ;
  - `bag` ;
  - `watch` ;
  - `sunglasses` ;
  - `wallet` ;
  - `belt` ;
  - `jewellery`.
- `canonical_category` est derive ensuite de facon deterministe pour le moteur outfit.
- Exemple attendu apres validation du mapping :
  - `articleType = "Tshirts"` -> `product_type_v0 = "tshirt"` -> `canonical_category = "top"`.
- Ne pas entrainer simultanement `articleType`, couleur ou usage en V0.

## Configuration des classes
- Fichier de configuration : `config/fashion_v1_classes.json`.
- Format :
  - `target`: `product_type_v0` ;
  - `source_column`: `articleType` ;
  - `status`: `validated_for_training` ;
  - `minimum_readable_images_per_class`: `450` ;
  - `product_type_mapping`: dictionnaire `product_type_v0 -> liste des articleType acceptes` ;
  - `canonical_mapping`: dictionnaire `product_type_v0 -> canonical_category`.
- Le mapping a ete valide apres audit reel du dataset image.
- `cap` est exclu de la V1 : seulement `283` images lisibles.

## Mapping metier candidat
- `tshirt`, `shirt`, `top` -> `top`.
- `jeans`, `trousers`, `shorts` -> `bottom`.
- `dress` -> `dress`.
- `outerwear` -> `outerwear`.
- `casual_shoes`, `sports_shoes`, `dress_shoes`, `sandals`, `flip_flops`, `heels` -> `shoes`.
- `bag` -> `bag`.
- `watch`, `sunglasses`, `wallet`, `belt`, `jewellery` -> `accessory`.

## Pipeline impose
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

## Tableau final attendu dans le notebook
Le notebook doit produire un tableau par `articleType` avec :

- `articleType` ;
- `proposed_product_type_v0` ;
- `proposed_canonical_category` ;
- `metadata_row_count` ;
- `present_image_count` ;
- `readable_image_count` ;
- `decision` : `garder` ou `exclure` ;
- `exclusion_reason`.

Depuis la validation V1, les `articleType` retenus doivent passer en `garder` si les images lisibles respectent le seuil minimal de `450`.

## Criteres de selection des classes
- V1.1 retient 20 classes `product_type_v0` visuellement coherentes.
- `bag`, `watch`, `sunglasses`, `wallet`, `belt` et `jewellery` restent des classes distinctes.
- `sandals`, `flip_flops` et `heels` restent separes de `casual_shoes`, `sports_shoes` et `dress_shoes`.
- Decision V1.1 : `articleType = "Flats"` est conserve dans le dataset mais mappe vers `product_type_v0 = "dress_shoes"`. La classe visible `flats` est retiree apres le premier entrainement, car elle avait un F1 test de `0.138`, un recall de `0.080`, et une confusion majoritaire vers `heels`.
- Exclure un `articleType` si :
  - categorie trop rare apres verification images lisibles ;
  - label ambigu ou incoherent avec les types produit V0 ;
  - lien faible avec le futur moteur outfit ;
  - trop grand taux d'images manquantes ou corrompues.
- Le seuil minimal retenu est `450` images lisibles par classe.

## Split
- Split stratifie par `product_type_v0`.
- Seed : `42`.
- Proposition future par defaut :
  - train : 70 % ;
  - validation : 15 % ;
  - test : 15 %.
- Le test ne doit etre utilise qu'apres selection finale de l'architecture et des hyperparametres.

## Architectures futures
- Variante 1 : CNN simple TensorFlow/Keras.
  - Normalisation image : `/255` uniquement.
- Variante 2 : transfer learning MobileNetV2.
  - Normalisation image : `tf.keras.applications.mobilenet_v2.preprocess_input` uniquement.
- Ne jamais appliquer `/255` puis `preprocess_input` sur la meme image.
- Le script `src/training/train_fashion_model_v1.py` compare `simple_cnn` et `mobilenet_v2` via `--architecture both`.
- La selection de l'architecture se fait uniquement sur validation.
- Le test est evalue une seule fois apres selection finale.

## Artefacts futurs attendus
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

`models/fashion_v1/` reste experimental. Un futur modele actif devra etre copie volontairement dans `models/fashion_active/` avec metadata promues.

## Resultat experimental V1.1
- Architecture selectionnee : `mobilenet_v2`.
- Test accuracy : `0.8748`.
- Test balanced accuracy : `0.8483`.
- Test macro F1 : `0.8526`.
- Test weighted F1 : `0.8725`.
- `flats` n'est plus une sortie visible ; `articleType = "Flats"` est mappe vers `dress_shoes`.
- Classe la plus fragile : `dress_shoes`, avec precision `0.729`, recall `0.456`, F1 `0.561`.
- Analyse d'abstention : seuil `0.90` selectionne sur validation ; test coverage `0.7083`, unknown rate `0.2917`, accuracy non-unknown `0.9695`, macro F1 non-unknown `0.9425`.
- Decision : V1.1 est promu localement dans `models/fashion_active/` avec abstention obligatoire sous `0.90`.

## Commandes Colab
Dry-run sans entrainement :

```bash
python -m src.training.train_fashion_model_v1 \
  --metadata-csv /content/fit-outfit-runtime/kaggle_downloads/myntradataset/styles.csv \
  --image-dir /content/fit-outfit-runtime/kaggle_downloads/images \
  --dry-run
```

Entrainement experimental :

```bash
python -m src.training.train_fashion_model_v1 \
  --metadata-csv /content/fit-outfit-runtime/kaggle_downloads/myntradataset/styles.csv \
  --image-dir /content/fit-outfit-runtime/kaggle_downloads/images \
  --architecture both \
  --epochs 12 \
  --batch-size 32 \
  --image-size 224
```

Analyse des seuils de confiance sans reentrainement :

```bash
python -m src.analysis.analyze_fashion_v1_abstention \
  --metadata-csv /content/fit-outfit-runtime/kaggle_downloads/myntradataset/styles.csv \
  --image-dir /content/fit-outfit-runtime/kaggle_downloads/images \
  --artifact-dir models/fashion_v1 \
  --output reports/fashion_v1_abstention.json
```

## Metriques attendues
- accuracy ;
- macro F1 ;
- balanced accuracy ;
- precision / recall / F1 par classe ;
- matrices de confusion brute et normalisee ;
- exemples de bonnes et mauvaises predictions.

## Risques dataset
- `articleType` peut etre trop detaille ou heterogene.
- Certaines classes peuvent etre trop rares pour une V0 stable.
- Des images peuvent manquer ou etre corrompues malgre la presence des lignes metadata.
- Certaines categories commerciales ou accessoires peuvent etre peu utiles au moteur outfit.
- Les couleurs et usages ne doivent pas etre melanges a la cible V0.

## Criteres de promotion Streamlit
- Metadata active obligatoires :
  - `model_status: "promoted"` ;
  - `promotable_to_streamlit: true`.
- Le modele doit etre dans `models/fashion_active/`, pas seulement dans `models/fashion_v1/`.
- Etat courant : Fashion V1.1 est copie localement dans `models/fashion_active/` avec `abstention_strategy.minimum_confidence: 0.90`.
- Les metriques doivent etre relues apres test final et documentees.
- Les classes predites doivent etre les `product_type_v0`; le mapping vers `canonical_category` doit etre deterministe et documente.
- Le seuil de confiance doit etre choisi sur validation uniquement et stocke dans `metadata.abstention_strategy.minimum_confidence` si le modele est promu.
- Sous ce seuil, le service image doit retourner `product_type: "unknown"` plutot qu'une classe ferme.
- Tant que ces conditions ne sont pas remplies, `image_service` reste en fallback simule.
