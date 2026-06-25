# Models

Ce dossier accueillera les modèles générés après entraînement :

- `fit_active/fit_model.keras` : seul emplacement lu par défaut par le service fit ;
- `fit_active/fit_preprocessor.joblib` : preprocessor du modèle fit promu ;
- `fit_active/fit_label_encoder.joblib` : encodeur du modèle fit promu ;
- `fit_active/metadata.json` : doit contenir `model_status: "promoted"` et `promotable_to_streamlit: true` ;
- `fit_v2/fit_model.keras` : modèle MLP ModCloth V2 expérimental ;
- `fit_v2/fit_preprocessor.joblib` : preprocessor tabulaire ModCloth V2 ;
- `fit_v2/fit_label_encoder.joblib` : encodeur des classes `small`, `fit`, `large` ;
- `fit_v2/metadata.json` : métadonnées V2 ; ce dossier reste expérimental même s'il contient des artefacts ;
- `fit_v2/metrics.json` : métriques validation/test et sélection d'expérience ;
- `fashion_model.keras` : modèle CNN entraîné sur Fashion Product Images Small ;
- `encoders/` : dossier conservé pour compatibilité avec les premiers essais.

Règle de promotion fail-closed :

- aucun modèle ne devient actif uniquement parce qu'il existe dans `models/` ;
- `fit_v2/` n'est jamais l'emplacement actif par défaut ;
- un modèle fit actif doit être copié volontairement dans `fit_active/` avec des metadata explicitement promues ;
- tout metadata absent, illisible ou incomplet doit refuser l'usage du modèle.

Les fichiers lourds de modèles ne sont pas versionnés par défaut.
