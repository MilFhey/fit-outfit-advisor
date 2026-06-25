# MODCLOTH_V3_EXPERIMENT_PLAN

## Decision V2
- V2 ameliore le baseline majoritaire sur le test :
  - macro F1 test : `0.357` vs baseline `0.271` ;
  - balanced accuracy test : `0.434` vs baseline `0.333` ;
  - recall `small` : `0.502` ;
  - recall `large` : `0.457`.
- V2 reste insuffisant pour produire un conseil utilisateur fiable :
  - accuracy : `0.385` ;
  - precision `small` : `0.209` ;
  - precision `large` : `0.228` ;
  - recall `fit` : `0.341`.
- Decision : V2 est conservable comme resultat academique et baseline ameliore, mais non promouvable vers Streamlit pour une recommandation ferme de taille.

## Garde-fous avant V3
- Aucun artefact V2 ne doit etre presente comme recommandation fiable dans Streamlit.
- `metadata.json` doit exposer `promotable_to_streamlit: false` et un statut explicite `experimental_only`.
- Le service d'inference doit refuser l'usage ferme d'un artefact experimental et revenir a un mode prudent.
- Le contrat d'inference ne doit contenir que les variables reellement apprises et raisonnablement demandables avant achat.
- `body_type` est exclu du contrat d'inference tant qu'il n'est pas justifie comme variable fiable et demandable dans le parcours utilisateur.

## Categories ModCloth
- Categories vestimentaires exploitables :
  - `tops`
  - `dresses`
  - `bottoms`
  - `outerwear`
  - `wedding`
- Categories commerciales ambiguës :
  - `new`
  - `sale`
- V3 doit evaluer separement :
  - le dataset complet ;
  - le dataset sans `new` et `sale` ;
  - le dataset limite aux categories vestimentaires explicites.

## Analyses V3 a produire avant modification d'architecture
1. Distribution globale des classes `fit`, `small`, `large`.
2. Distribution des classes par categorie ModCloth.
3. Performance par categorie sur validation et test final.
4. Performance lorsque `new` et `sale` sont exclus.
5. Performance en conservant uniquement les categories vestimentaires explicites.
6. Verification du sens de la colonne `size` :
   - nom exact de colonne ;
   - exemples de valeurs ;
   - distribution ;
   - correlation descriptive avec `fit` ;
   - confirmation que `item_size_order` represente la taille de l'article, pas une taille utilisateur post-achat.
7. Verification de la transformation `item_size_order` :
   - taux de valeurs invalides ;
   - taux de valeurs manquantes ;
   - distribution apres parsing ;
   - effet de l'imputation.

## Variables candidates et risque de fuite
Les variables ne sont retenues que si elles peuvent etre demandees raisonnablement a l'utilisateur avant achat.

| Variable | Statut V3 | Justification |
| --- | --- | --- |
| `height` | Candidate | Mesure utilisateur simple, demandable avant achat. |
| `bust` | Candidate prudente | Demandable, mais sensible ; verifier disponibilite et taux de manque. |
| `hips` | Candidate prudente | Demandable, mais sensible ; verifier disponibilite et taux de manque. |
| `waist` | Candidate prudente | Demandable, mais sensible ; verifier disponibilite et taux de manque. |
| `bra size` | Candidate prudente | Demandable pour certains vetements, mais sensible et non universelle. |
| `cup size` | Candidate prudente | Demandable pour certains vetements, mais sensible et non universelle. |
| `body type` | Non retenue par defaut | Categorie subjective, difficile a demander proprement, contrat V2 a corriger. |
| `rating`, `review_text`, `quality`, `rented for` | Exclues | Disponibles apres experience ou dependantes du review ; risque de fuite ou hors parcours achat. |

Pour chaque mensuration retenue, V3 doit ajouter un indicateur binaire de valeur manquante, par exemple `height_cm_missing`, `bust_missing`, `waist_missing`.

## Experiences minimales a comparer
- Baseline majoritaire.
- Regression logistique.
- MLP TensorFlow sans ponderation.
- MLP TensorFlow avec `class_weight`.

La selection entre experiences doit rester faite exclusivement sur le jeu de validation. Le jeu de test ne doit etre utilise qu'une fois apres selection finale.

## Strategie d'abstention
- Ajouter une classe de sortie service `uncertain` lorsque la confiance est faible.
- Ne jamais transformer une prediction faible confiance en recommandation ferme `small` ou `large`.
- Prevoir une formulation prudente pour le futur service :
  - "Le modele n'est pas assez confiant pour recommander une taille differente."
  - "Utilise cette sortie comme signal exploratoire, pas comme conseil de taille ferme."
- V3 doit evaluer plusieurs seuils de confiance sur validation :
  - couverture ;
  - accuracy sur predictions non abstention ;
  - macro F1 sur predictions non abstention ;
  - taux d'abstention par classe.

## Livrables attendus V3
- Un rapport de distribution et performance par categorie.
- Un tableau comparatif des experiences minimales.
- Une analyse avant/apres exclusion de `new` et `sale`.
- Une analyse des categories vestimentaires explicites uniquement.
- Une justification des variables retenues et exclues.
- Des metadonnees avec :
  - `promotable_to_streamlit: false` tant que les seuils metier ne sont pas atteints ;
  - `model_status: "experimental_only"` ;
  - `inference_contract` limite aux champs reels et demandables.

## Preparation Colab sans nouvel entrainement
- Ouvrir `notebooks/01_train_fit_model_colab.ipynb` uniquement pour preparer l'environnement et telecharger le dataset.
- Executer les cellules jusqu'a l'inspection du dataset incluse :
  - montage Google Drive ;
  - clone ou mise a jour du repo ;
  - installation des dependances ;
  - creation des dossiers temporaires ;
  - chargement du secret Kaggle ;
  - telechargement ModCloth ;
  - detection du fichier dataset ;
  - `df.head()`, `df.columns`, `df.shape`, valeurs manquantes.
- S'arreter avant la cellule `Lancer l'entrainement ModCloth V2`.
- Ne pas copier d'artefacts experimentaux vers `models/fit_active/`.
- Garder les artefacts V2 dans `models/fit_v2/` ou dans Drive sous `artifacts/modcloth_fit_v2/`.
- Pour une future analyse V3, utiliser le dataset detecte dans Colab comme entree de scripts/sections d'analyse descriptives, sans appeler `src.training.train_fit_model`.
- Les analyses descriptives autorisees avant entrainement V3 sont :
  - distributions de classes ;
  - distributions par categorie ;
  - taux de valeurs manquantes des mensurations candidates ;
  - diagnostics de `size` et `item_size_order` ;
  - comptages avant/apres exclusion de `new` et `sale`.
