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
- Cible apprise par le CNN : `canonical_category`.
- Colonne source : `articleType`.
- Le modele doit predire directement une categorie canonique applicative :
  - `top` ;
  - `bottom` ;
  - `dress` ;
  - `shoes` ;
  - `outerwear` ;
  - `accessory`.
- Exemple attendu apres validation du mapping :
  - `articleType = "Tshirts"` -> `canonical_category = "top"`.
- Ne pas entrainer simultanement `articleType`, couleur ou usage en V0.

## Configuration des classes
- Fichier de configuration : `config/fashion_v1_classes.json`.
- Format :
  - `target`: `canonical_category` ;
  - `source_column`: `articleType` ;
  - `status`: `draft_requires_dataset_inspection` tant que les classes ne sont pas validees ;
  - `minimum_readable_images_per_class`: `null` tant que le seuil n'est pas choisi ;
  - `mapping`: dictionnaire `canonical_category -> liste des articleType acceptes`.
- Les listes du mapping restent volontairement vides avant inspection reelle du dataset.
- Aucune classe V0 ne doit etre consideree comme retenue tant que :
  - le tableau final du notebook n'a pas ete relu ;
  - les `articleType` acceptes n'ont pas ete inscrits dans `config/fashion_v1_classes.json` ;
  - un seuil minimal d'images lisibles par classe n'a pas ete documente.

## Pipeline impose
```text
styles.csv
-> mapping articleType vers canonical_category
-> exclusion labels non retenus
-> verification fichier image present et lisible
-> comptage final par classe
-> seuil minimal documente par classe
-> split stratifie
```

## Tableau final attendu dans le notebook
Le notebook doit produire un tableau par `articleType` avec :

- `articleType` ;
- `proposed_canonical_category` ;
- `metadata_row_count` ;
- `present_image_count` ;
- `readable_image_count` ;
- `decision` : `garder` ou `exclure` ;
- `exclusion_reason`.

Tant que la configuration reste en brouillon, la decision peut rester `exclure` meme si une categorie canonique est proposee pour revue.

## Criteres de selection des classes
- Viser 5 a 8 classes canoniques utiles au produit.
- Exclure un `articleType` si :
  - categorie trop rare apres verification images lisibles ;
  - label ambigu ou incoherent avec les categories canoniques ;
  - lien faible avec le futur moteur outfit ;
  - trop grand taux d'images manquantes ou corrompues.
- Le seuil minimal d'images lisibles par classe doit etre choisi apres inspection et inscrit dans `config/fashion_v1_classes.json`.

## Split
- Split stratifie par `canonical_category`.
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
- Les classes predites doivent etre les categories canoniques attendues par l'application.
- Tant que ces conditions ne sont pas remplies, `image_service` reste en fallback simule.
