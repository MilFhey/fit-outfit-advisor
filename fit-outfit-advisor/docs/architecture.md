# Architecture MVP

```mermaid
flowchart TD
    U[Utilisateur] --> S[Streamlit App]

    S --> IS[Image Service]
    S --> FS[Fit Service]
    S --> OS[Outfit Service]
    S --> AS[Advice Service]

    IS --> IM[Simulation puis CNN Fashion]
    FS --> FM[Simulation puis MLP ModCloth]
    OS --> MAP[Mappings catégories/couleurs]

    IM --> AS
    FM --> AS
    OS --> AS

    AS --> R[Conseil final]
```

## Principe

Streamlit ne doit pas contenir la logique métier. Il doit uniquement :

1. collecter les entrées utilisateur ;
2. appeler les services ;
3. afficher les résultats.

Les services doivent être réutilisables plus tard par FastAPI.

## Future API cible

```text
POST /predict/image
POST /predict/fit
POST /recommend/outfit
POST /advice/final
```
