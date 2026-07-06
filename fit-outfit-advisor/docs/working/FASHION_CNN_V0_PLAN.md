# FASHION_CNN_V0_PLAN - Fashion Product Images Small

## Statut
- Nouvelle priorite apres cloture du module ModCloth comme experimentation academique.
- Aucun entrainement image local ne doit etre lance tant que le dataset et les classes V0 ne sont pas inspectes.
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
  - `flats` ;
  - `bag` ;
  - `watch` ;
  - `sunglasses` ;
  - `cap` ;
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
  - `status`: `draft_requires_dataset_inspection` tant que les classes ne sont pas validees ;
  - `minimum_readable_images_per_class`: `null` tant que le seuil n'est pas choisi ;
  - `product_type_mapping`: dictionnaire `product_type_v0 -> liste des articleType acceptes` ;
  - `canonical_mapping`: dictionnaire `product_type_v0 -> canonical_category`.
- Le mapping reste en brouillon avant inspection reelle du dataset.
- Aucune classe V0 ne doit etre consideree comme retenue tant que :
  - le tableau final du notebook n'a pas ete relu ;
  - les `articleType` acceptes n'ont pas ete inscrits dans `config/fashion_v1_classes.json` ;
  - un seuil minimal d'images lisibles par classe n'a pas ete documente.

## Mapping metier candidat
- `tshirt`, `shirt`, `top` -> `top`.
- `jeans`, `trousers`, `shorts` -> `bottom`.
- `dress` -> `dress`.
- `outerwear` -> `outerwear`.
- `casual_shoes`, `sports_shoes`, `dress_shoes`, `sandals`, `flip_flops`, `heels`, `flats` -> `shoes`.
- `bag` -> `bag`.
- `watch`, `sunglasses`, `cap`, `wallet`, `belt`, `jewellery` -> `accessory`.

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

Tant que la configuration reste en brouillon, la decision peut rester `exclure` meme si un `product_type_v0` est propose pour revue.

## Criteres de selection des classes
- Viser environ 10 a 20 classes `product_type_v0` visuellement coherentes, selon les effectifs lisibles reels.
- `bag`, `watch`, `sunglasses`, `cap`, `wallet`, `belt` et `jewellery` peuvent rester des classes distinctes si elles respectent le seuil d'images lisibles.
- `sandals`, `flip_flops`, `heels` et `flats` peuvent rester separes de `casual_shoes`, `sports_shoes` et `dress_shoes` si les effectifs restent suffisants.
- Exclure un `articleType` si :
  - categorie trop rare apres verification images lisibles ;
  - label ambigu ou incoherent avec les types produit V0 ;
  - lien faible avec le futur moteur outfit ;
  - trop grand taux d'images manquantes ou corrompues.
- Le seuil minimal d'images lisibles par classe doit etre choisi apres inspection et inscrit dans `config/fashion_v1_classes.json`.

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
- Les metriques doivent etre relues apres test final et documentees.
- Les classes predites doivent etre les `product_type_v0`; le mapping vers `canonical_category` doit etre deterministe et documente.
- Tant que ces conditions ne sont pas remplies, `image_service` reste en fallback simule.
