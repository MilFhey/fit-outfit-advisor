# Models

Ce dossier accueillera les modèles générés après entraînement :

- `fit_v2/fit_model.keras` : modèle MLP ModCloth V2 expérimental ;
- `fit_v2/fit_preprocessor.joblib` : preprocessor tabulaire ModCloth V2 ;
- `fit_v2/fit_label_encoder.joblib` : encodeur des classes `small`, `fit`, `large` ;
- `fit_v2/metadata.json` : métadonnées, contrat d'inférence et statut de promotion ;
- `fit_v2/metrics.json` : métriques validation/test et sélection d'expérience ;
- `fashion_model.keras` : modèle CNN entraîné sur Fashion Product Images Small ;
- `encoders/` : dossier conservé pour compatibilité avec les premiers essais.

Les fichiers lourds de modèles ne sont pas versionnés par défaut.
