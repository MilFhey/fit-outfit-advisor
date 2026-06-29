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

## Observations Colab reelles avant entrainement
- Dataset inspecte dans Colab avant toute execution d'entrainement.
- Shape observee : `(82790, 18)`.
- Colonnes observees :
  - `item_id`
  - `waist`
  - `size`
  - `quality`
  - `cup size`
  - `hips`
  - `bra size`
  - `category`
  - `bust`
  - `height`
  - `user_name`
  - `length`
  - `fit`
  - `user_id`
  - `shoe size`
  - `shoe width`
  - `review_summary`
  - `review_text`

### Valeurs manquantes observees
| Colonne | Missing | Missing approx. | Decision V3 provisoire |
| --- | ---: | ---: | --- |
| `height` | 1107 | 1.3% | Retenir : mesure pre-achat simple, peu manquante. |
| `hips` | 26726 | 32.3% | Candidate forte avec indicateur `hips_missing`. |
| `bra size` | 6018 | 7.3% | Candidate prudente avec indicateur, utile selon type de vetement. |
| `cup size` | 6255 | 7.6% | Candidate prudente avec indicateur, parser les valeurs composees. |
| `waist` | 79908 | 96.5% | Ne pas retenir dans V3 initial sauf analyse dediee : trop manquante. |
| `bust` | 70936 | 85.7% | Ne pas retenir dans V3 initial sauf analyse dediee : trop manquante. |
| `shoe size` | 54875 | 66.3% | Exclure du pipeline vetements general ; eventuellement analyse chaussures separee. |
| `shoe width` | 64183 | 77.5% | Exclure du pipeline vetements general ; eventuellement analyse chaussures separee. |
| `quality` | 68 | 0.1% | Exclure : information post-achat / review. |
| `length` | 35 | 0.0% | Exclure : information post-achat sur le ressenti du vetement. |
| `review_summary` | 6732 | 8.1% | Exclure : texte post-achat, fuite de donnees. |
| `review_text` | 6732 | 8.1% | Exclure : texte post-achat, fuite de donnees. |

### Decisions provisoires sur les colonnes
- Cible :
  - `fit`.
- Features candidates pre-achat pour V3 descriptif :
  - `size`
  - `category`
  - `height`
  - `hips`
  - `bra size`
  - `cup size`
- Features candidates mais a exclure du premier V3 robuste :
  - `waist`, car trop manquante ;
  - `bust`, car trop manquante ;
  - `shoe size`, `shoe width`, car hors pipeline vetements general et tres manquantes.
- Colonnes a exclure pour eviter fuite ou memorisation :
  - `quality`
  - `length`
  - `review_summary`
  - `review_text`
  - `user_name`
  - `user_id`
  - `item_id` dans le modele initial, car identifiant haute cardinalite pouvant favoriser la memorisation produit.
- `size` doit encore etre auditee : les valeurs observees comme `7`, `13`, `18`, `21` suggerent une echelle interne ou encodee, pas forcement une taille utilisateur directement interpretable.

### Distribution reelle des classes
| Classe | Count | Proportion |
| --- | ---: | ---: |
| `fit` | 56757 | 68.56% |
| `large` | 13059 | 15.77% |
| `small` | 12974 | 15.67% |

- Baseline majoritaire complet attendu : predire toujours `fit`, accuracy environ `0.686`.
- Le desequilibre est important mais pas extreme : les classes `large` et `small` sont proches en volume.
- L'accuracy seule reste inadaptee comme metrique de selection.

### Distribution reelle par categorie
| Categorie | Count | Proportion |
| --- | ---: | ---: |
| `new` | 21488 | 25.95% |
| `tops` | 20364 | 24.60% |
| `dresses` | 18650 | 22.53% |
| `bottoms` | 15266 | 18.44% |
| `outerwear` | 4223 | 5.10% |
| `sale` | 2524 | 3.05% |
| `wedding` | 275 | 0.33% |

- Les categories natives observees sont exactement :
  - categories vestimentaires explicites : `tops`, `dresses`, `bottoms`, `outerwear`, `wedding` ;
  - categories commerciales ambigues : `new`, `sale`.
- Exclure `new` et `sale` donne le meme sous-ensemble que conserver uniquement les categories explicites :
  - dataset complet : `(82790, 18)` ;
  - sans `new/sale` : `(58778, 18)` ;
  - categories explicites seulement : `(58778, 18)`.
- L'exclusion de `new/sale` retire `24012` lignes, soit environ `29.0%` du dataset.

### Distribution `fit` apres exclusion de `new/sale`
| Sous-ensemble | `fit` | `large` | `small` |
| --- | ---: | ---: | ---: |
| Dataset complet | 0.686 | 0.158 | 0.157 |
| Sans `new/sale` | 0.693 | 0.164 | 0.143 |
| Categories explicites | 0.693 | 0.164 | 0.143 |

- Supprimer `new/sale` n'equilibre pas les classes.
- Le taux de `small` diminue de `15.7%` a `14.3%`.
- Le taux de `large` augmente legerement de `15.8%` a `16.4%`.
- Le baseline majoritaire devient legerement plus fort sur le sous-ensemble explicite : environ `0.693`.

### Signal categorie vs cible
| Categorie | `fit` | `large` | `small` | Lecture |
| --- | ---: | ---: | ---: | --- |
| `bottoms` | 0.698 | 0.135 | 0.167 | proche global, un peu plus `small`. |
| `dresses` | 0.728 | 0.137 | 0.135 | plus favorable a `fit`. |
| `new` | 0.671 | 0.139 | 0.190 | categorie commerciale, davantage de `small`. |
| `outerwear` | 0.661 | 0.193 | 0.145 | davantage de `large`. |
| `sale` | 0.630 | 0.172 | 0.198 | categorie commerciale, davantage d'erreurs de fit. |
| `tops` | 0.663 | 0.205 | 0.132 | davantage de `large`. |
| `wedding` | 0.796 | 0.156 | 0.047 | tres faible volume, a traiter avec prudence. |

- `category` porte un signal reel, mais il n'est pas suffisant seul.
- `wedding` est trop petit pour une evaluation robuste par categorie.
- `tops` et `outerwear` semblent plus sujets a `large`.
- `new` et `sale` sont ambigus commercialement mais contiennent un signal de distribution ; V3 doit comparer avec et sans eux.

### Premiers constats sur `size`
- `size` n'a pas de valeur manquante.
- Les exemples observes (`3`, `5`, `7`, `9`, `11`, `13`, `15`, `18`, `21`, `24`, `27`, `30`, `33`) suggerent une echelle ordonnee ModCloth.
- `size` peut probablement etre conserve comme `item_size_order`, mais son sens metier doit etre documente :
  - il semble decrire la taille de l'article commande ;
  - il ne doit pas etre confondu avec une taille habituelle utilisateur ;
  - il reste demandable avant achat si l'utilisateur choisit la taille de l'article a evaluer.

### Diagnostic approfondi `height` et `size`
#### `height_cm` par cible
| Classe | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit` | 56008 | 165.35 | 7.21 | 91.44 | 160.02 | 165.10 | 170.18 | 241.30 |
| `large` | 12870 | 165.51 | 7.23 | 91.44 | 160.02 | 165.10 | 170.18 | 231.14 |
| `small` | 12805 | 165.97 | 7.39 | 91.44 | 160.02 | 165.10 | 170.18 | 241.30 |

- `height_cm` varie tres peu entre classes.
- Les valeurs extremes (`91.44`, `231+`, `241.30`) doivent etre auditees comme outliers ou erreurs de parsing/saisie.
- Decision V3 : conserver `height_cm` avec imputation et indicateur manquant, mais ne pas attendre un fort gain seul.

#### `size` par cible
| Classe | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit` | 56757 | 12.04 | 7.90 | 0 | 8 | 12 | 15 | 38 |
| `large` | 13059 | 14.24 | 9.27 | 0 | 8 | 12 | 20 | 38 |
| `small` | 12974 | 13.81 | 8.47 | 0 | 8 | 12 | 15 | 38 |

- `large` et `small` ont en moyenne un `size` plus eleve que `fit`.
- L'effet varie selon la categorie :
  - `outerwear`: `large` moyen `16.40`, `fit` moyen `11.69`, `small` moyen `11.82` ;
  - `tops`: `small` moyen `15.31`, `large` moyen `13.35`, `fit` moyen `12.68` ;
  - `bottoms`: `small` moyen `14.60`, `large` moyen `13.27`, `fit` moyen `11.76`.
- Le sens de `size` n'est donc pas un simple "plus grand = plus large" universel.
- V3 doit evaluer des interactions `category x size` ou au minimum comparer les performances par categorie.

#### Correlations numeriques observees
| Variable A | Variable B | Correlation |
| --- | --- | ---: |
| `size` | `height_cm` | 0.207 |
| `size` | `hips` | 0.747 |
| `size` | `bra size` | 0.788 |
| `height_cm` | `hips` | 0.191 |
| `height_cm` | `bra size` | 0.179 |
| `hips` | `bra size` | 0.671 |

- `size` est fortement correle a `hips` et `bra size`.
- Interpretation probable : `size` encode la taille de l'article choisi, mais cette taille est aussi fortement liee a la morphologie de l'utilisatrice.
- Risque methodologique : utiliser `size` sans mensurations peut masquer un signal morphologique ; utiliser `size` avec mensurations peut creer une forte redondance.
- Decision V3 : conserver `size` comme variable demandable avant achat, mais documenter explicitement qu'elle represente la taille article choisie et pas la taille habituelle utilisateur.
- Comparaison V3 recommandee :
  - modele sans mensurations : `size`, `category`, `height`;
  - modele avec mensurations retenues : `size`, `category`, `height`, `hips`, `bra size`, `cup size`;
  - evaluation par categorie pour verifier si le signal change selon `tops`, `bottoms`, `dresses`, `outerwear`.

#### Completeness des feature sets candidats
| Sous-ensemble | Lignes apres `dropna` | Couverture | `fit` | `large` | `small` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `size + category + height` | 81683 | 98.66% | 0.686 | 0.158 | 0.157 |
| `+ hips` | 55828 | 67.43% | 0.687 | 0.162 | 0.152 |
| `+ bra size + cup size` | 75709 | 91.45% | 0.685 | 0.158 | 0.157 |
| `+ hips + bra size + cup size` | 54345 | 65.64% | 0.687 | 0.161 | 0.151 |

- Ajouter `hips` et supprimer les lignes manquantes ferait perdre environ un tiers du dataset.
- Ajouter `bra size` + `cup size` conserve plus de `91%` des lignes.
- Les distributions de `fit` restent presque identiques dans tous les sous-ensembles.
- Decision V3 : ne pas faire de `dropna` global sur les mensurations candidates.
- Strategie recommandee :
  - imputer les valeurs numeriques sur le train seulement ;
  - ajouter des indicateurs de manque (`height_cm_missing`, `hips_missing`, `bra_size_missing`, `cup_size_missing`) ;
  - comparer explicitement un modele minimal et un modele enrichi.
- Le subset avec mensurations completes peut servir uniquement a une analyse de sensibilite, pas comme dataset principal.

#### Derniers controles pre-entrainement
##### Cardinalites observees
| Colonne | Cardinalite | Notes |
| --- | ---: | --- |
| `size` | 29 | Valeurs de `0` a `38`, avec deux valeurs tres rares `25` et `31`. |
| `category` | 7 | Aucune categorie inconnue hors groupes prevus. |
| `cup size` | 13 | Inclut `NaN` et valeurs composees (`dd/e`, `ddd/f`, `dddd/g`). |
| `bra size` | 12 | Valeurs de `28` a `48`, plus `NaN`. |
| `hips` | 32 | Valeurs de `30` a `60`, plus `NaN`. |
| `height` | 42 | Inclut des outliers tres bas et tres hauts. |

##### Categories
- Categories inconnues : aucune (`set()`).
- Categories explicites :
  - `tops`: 20364 ;
  - `dresses`: 18650 ;
  - `bottoms`: 15266 ;
  - `outerwear`: 4223 ;
  - `wedding`: 275.
- Categories commerciales :
  - `new`: 21488 ;
  - `sale`: 2524.

##### Outliers `height_cm`
- `height_cm < 130`: 28 lignes.
- `height_cm > 210`: 29 lignes.
- Exemples bas : `3ft`, `3ft 2in`, `3ft 4in`, `3ft 6in`, `4ft 2in`.
- Exemples hauts : `7ft 3in`, `7ft 5in`, `7ft 6in`, `7ft 7in`, `7ft 11in`.
- Decision V3 : parser `height`, puis mettre `height_cm` hors plage plausible en valeur manquante avant imputation.
- Plage plausible recommandee pour V3 initial : `130 <= height_cm <= 210`.
- Ajouter/conserver `height_cm_missing` apres ce nettoyage, afin que les outliers convertis en manquants soient signales.

##### Valeurs extremes numeriques
- `size` :
  - count `82790`, mean `12.66`, min `0`, max `38`.
  - `size=0` existe mais reste rare (`31` lignes).
  - Decision V3 : conserver `size=0` pour l'instant comme valeur native observee, sauf preuve qu'il s'agit d'une erreur de codage.
- `hips` :
  - count non manquant `56064`, mean `40.36`, min `30`, max `60`.
  - Decision V3 : valeurs plausibles pour le dataset ; conserver avec imputation et `hips_missing`.
- `bra size` :
  - count non manquant `76772`, mean `35.97`, min `28`, max `48`.
  - Decision V3 : valeurs plausibles ; conserver avec imputation et `bra_size_missing`.

##### Feu vert pre-entrainement V3
- Les controles pre-entrainement ne bloquent pas le lancement d'une experience V3.
- Le training V3 doit rester experimental :
  - ecrire dans un dossier separe, par exemple `models/fit_v3/` ;
  - `model_status: "experimental_only"` ;
  - `promotable_to_streamlit: false` ;
  - aucune copie automatique vers `models/fit_active/`.
- Nettoyage minimal requis avant training :
  - normaliser les noms de colonnes (`cup size` -> `cup_size`, `bra size` -> `bra_size`) ;
  - parser `height` en `height_cm` ;
  - convertir `height_cm` hors `[130, 210]` en manquant ;
  - creer les indicateurs `height_cm_missing`, `hips_missing`, `bra_size_missing`, `cup_size_missing` ;
  - conserver `cup_size` comme categoriel, sans parsing arbitraire ;
  - imputer numeriques et categoriels via un preprocessor ajuste sur train uniquement ;
  - exclure les colonnes post-achat et identifiants.

### Signal mensurations vs cible
#### `hips`
| Classe | Count non manquant | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit` | 38503 | 40.05 | 5.70 | 30 | 36 | 39 | 43 | 60 |
| `large` | 9062 | 41.00 | 6.14 | 30 | 36 | 40 | 45 | 60 |
| `small` | 8499 | 41.05 | 5.95 | 30 | 36 | 40 | 44 | 60 |

- `hips` semble porter un signal leger : les classes `large` et `small` ont une moyenne et une mediane un peu plus elevees que `fit`.
- Le signal est utile mais probablement faible seul, avec fort recouvrement des distributions.
- Comme `hips` manque dans `32.3%` des lignes, V3 doit conserver `hips_missing`.

#### `bra size`
| Classe | Count non manquant | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit` | 52601 | 35.80 | 3.15 | 28 | 34 | 36 | 38 | 48 |
| `large` | 12141 | 36.27 | 3.41 | 28 | 34 | 36 | 38 | 48 |
| `small` | 12030 | 36.43 | 3.30 | 28 | 34 | 36 | 38 | 48 |

- `bra size` a peu de valeurs manquantes et un faible signal directionnel.
- Les distributions restent tres superposees ; ne pas surinterpreter en regle metier directe.
- V3 doit l'evaluer comme variable candidate avec `bra_size_missing`.

#### `waist`
| Classe | Count non manquant | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit` | 1725 | 31.06 | 5.23 | 20 | 28 | 30 | 34.00 | 50 |
| `large` | 606 | 31.89 | 5.41 | 20 | 28 | 31 | 35.75 | 50 |
| `small` | 551 | 31.50 | 5.37 | 22 | 28 | 30 | 34.00 | 50 |

- `waist` montre un signal possible mais le taux de valeurs manquantes est trop eleve (`96.5%`).
- Decision maintenue : exclure du V3 initial robuste, sauf analyse dediee sur sous-population.

#### `cup size`
| Cup size | `fit` | `large` | `small` |
| --- | ---: | ---: | ---: |
| `a` | 0.726 | 0.157 | 0.117 |
| `aa` | 0.726 | 0.159 | 0.115 |
| `b` | 0.720 | 0.142 | 0.138 |
| `c` | 0.690 | 0.152 | 0.158 |
| `d` | 0.681 | 0.155 | 0.164 |
| `dd/e` | 0.669 | 0.164 | 0.167 |
| `ddd/f` | 0.637 | 0.184 | 0.179 |
| `dddd/g` | 0.625 | 0.194 | 0.181 |
| `h` | 0.606 | 0.208 | 0.186 |
| `i` | 0.646 | 0.201 | 0.153 |
| `j` | 0.615 | 0.240 | 0.145 |
| `k` | 0.580 | 0.202 | 0.218 |

- `cup size` contient un signal visible : plus la cup augmente, plus la proportion de `fit` tend a diminuer et plus `large`/`small` augmentent.
- Les categories composees (`dd/e`, `ddd/f`, `dddd/g`) doivent etre conservees ou mappees explicitement, pas parsees au hasard.
- V3 doit comparer au moins deux encodages :
  - categoriel brut avec imputation `missing` ;
  - ordre manuel documente (`aa`, `a`, `b`, `c`, `d`, `dd/e`, `ddd/f`, `dddd/g`, `h`, `i`, `j`, `k`) si l'ordre est conserve.

### Decision provisoire feature set V3
- V3 descriptif/modelisation initiale peut tester :
  - `item_size_order`
  - `category`
  - `height_cm`
  - `height_cm_missing`
  - `hips`
  - `hips_missing`
  - `bra_size`
  - `bra_size_missing`
  - `cup_size`
  - `cup_size_missing`
- A exclure du premier V3 robuste :
  - `waist`, `bust`, `shoe size`, `shoe width`, `quality`, `length`, `review_summary`, `review_text`, `user_id`, `user_name`, `item_id`.

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
