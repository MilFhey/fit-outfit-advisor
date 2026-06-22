import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.services.image_service import predict_image
from src.services.fit_service import predict_fit
from src.services.outfit_service import recommend_outfit
from src.services.advice_service import generate_advice


st.set_page_config(
    page_title="Fit & Outfit Advisor",
    page_icon="shirt",
    layout="wide",
)

st.title("Fit & Outfit Advisor")
st.caption("Prototype MVP Streamlit - services modulaires, modèles TensorFlow intégrés progressivement.")

with st.expander("Objectif du prototype", expanded=False):
    st.write(
        "Ce premier prototype valide le parcours utilisateur avant l'entraînement des modèles : "
        "upload image, profil utilisateur, prédiction de fit, suggestion de tenue et conseil final."
    )

col1, col2 = st.columns(2)

with col1:
    st.header("1. Image du vêtement")
    uploaded_file = st.file_uploader(
        "Ajoute une image produit",
        type=["jpg", "jpeg", "png"],
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_container_width=True)
    else:
        st.info("Ajoute une image pour simuler la reconnaissance du vêtement.")

with col2:
    st.header("2. Profil utilisateur")

    height = st.number_input("Taille utilisateur (cm)", min_value=120, max_value=220, value=175)
    weight = st.number_input("Poids utilisateur (kg)", min_value=35, max_value=180, value=75)
    usual_size = st.selectbox("Taille habituelle", ["XS", "S", "M", "L", "XL", "XXL"], index=2)
    context = st.selectbox("Contexte", ["casual", "travail", "soirée", "sport"])
    color = st.selectbox("Couleur principale", ["noir", "blanc", "bleu", "rouge", "beige", "vert", "bordeaux"])

st.header("3. Caractéristiques du vêtement")
col3, col4 = st.columns(2)

with col3:
    item_size = st.selectbox("Taille du vêtement", ["XS", "S", "M", "L", "XL", "XXL"], index=2)

with col4:
    brand = st.text_input("Marque", value="Marque inconnue")

st.divider()

if st.button("Analyser la tenue", type="primary"):
    user_profile = {
        "height_cm": height,
        "weight_kg": weight,
        "usual_size": usual_size,
    }

    item_features = {
        "item_size": item_size,
        "brand": brand,
        "color": color,
    }

    image_result = predict_image(image)
    fit_result = predict_fit(user_profile, item_features)
    outfit_result = recommend_outfit(
        image_result["common_category"],
        context,
        color,
    )
    advice_result = generate_advice(
        image_result,
        fit_result,
        outfit_result,
        context,
    )

    res1, res2, res3 = st.columns(3)

    with res1:
        st.subheader("Vêtement")
        st.metric("Détecté", image_result.get("predicted_class", "unknown"))
        st.write(f"Catégorie commune : **{image_result.get('common_category', 'unknown')}**")
        st.write(f"Confiance image : **{float(image_result.get('confidence', 0.0)):.0%}**")

    with res2:
        st.subheader("Taille")
        st.metric("Prédiction du fit", fit_result.get("fit_prediction", "unknown"))
        st.write(f"Confiance fit : **{float(fit_result.get('confidence', 0.0)):.0%}**")
        st.write(f"Niveau de risque : **{fit_result.get('risk_level', 'unknown')}**")

    with res3:
        st.subheader("Tenue")
        st.metric("Compatibilité", f"{float(outfit_result.get('compatibility_score', 0.0)):.0%}")
        st.write("Pièces compatibles :")
        st.write(", ".join(outfit_result.get("compatible_items", [])) or "Aucune suggestion")
        st.write("Couleurs compatibles :")
        st.write(", ".join(outfit_result.get("compatible_colors", [])) or "Aucune suggestion")

    st.subheader("Conseil final")
    st.success(advice_result.get("advice", "Conseil indisponible."))

    for warning in advice_result.get("warnings", []):
        st.warning(warning)

    with st.expander("Détails techniques", expanded=False):
        st.json(
            {
                "image": image_result,
                "fit": fit_result,
                "outfit": outfit_result,
                "advice": advice_result,
            }
        )

    st.info(
        "Note pédagogique : le service fit utilise le modèle TensorFlow si les artefacts sont présents, "
        "sinon il revient automatiquement au fallback simulé."
    )
else:
    st.warning("Renseigne les informations puis clique sur 'Analyser la tenue'.")
