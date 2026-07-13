import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.services.image_service import predict_image
from src.services.fit_service import predict_fit
from src.services.outfit_v2_service import (
    evaluate_outfit_images,
    recommend_associations_from_image,
)
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
        "Ce prototype combine reconnaissance image Fashion V1, compatibilite outfit V0/V2, "
        "regles couleur et fallback fail-closed."
    )

context = st.selectbox("Contexte", ["casual", "travail", "soirée", "sport"])
single_tab, outfit_tab = st.tabs(["Associer une pièce", "Evaluer une tenue"])

with single_tab:
    left, right = st.columns(2)
    with left:
        st.header("Image du vêtement")
        uploaded_file = st.file_uploader(
        "Ajoute une image produit",
        type=["jpg", "jpeg", "png"],
        key="single_item_upload",
    )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image chargée", use_container_width=True)
        else:
            st.info("Ajoute une image pour obtenir des associations.")

    with right:
        st.header("Profil et taille")
        height = st.number_input("Taille utilisateur (cm)", min_value=120, max_value=220, value=175)
        weight = st.number_input("Poids utilisateur (kg)", min_value=35, max_value=180, value=75)
        usual_size = st.selectbox("Taille habituelle", ["XS", "S", "M", "L", "XL", "XXL"], index=2)
        item_size = st.selectbox("Taille du vêtement", ["XS", "S", "M", "L", "XL", "XXL"], index=2)
        brand = st.text_input("Marque", value="Marque inconnue")

    if st.button("Proposer des associations", type="primary"):
        user_profile = {
            "height_cm": height,
            "weight_kg": weight,
            "usual_size": usual_size,
        }
        use_real_image_model = image is not None
        image_result = predict_image(image, use_real_model=use_real_image_model)
        outfit_result = recommend_associations_from_image(
            image,
            context,
            use_real_image_model=use_real_image_model,
        )
        color = (
            outfit_result.get("detected_item", {}).get("color_label")
            or outfit_result.get("compatible_colors", ["noir"])[0]
        )
        fit_result = predict_fit(
            user_profile,
            {"item_size": item_size, "brand": brand, "color": color},
            use_real_model=True,
        )
        advice_result = generate_advice(image_result, fit_result, outfit_result, context)

        res1, res2, res3 = st.columns(3)
        with res1:
            st.subheader("Vêtement")
            detected_item = outfit_result.get("detected_item", {})
            st.metric("Détecté", detected_item.get("product_type", image_result.get("predicted_class", "unknown")))
            st.write(f"Catégorie : **{detected_item.get('canonical_category', image_result.get('common_category', 'unknown'))}**")
            st.write(f"Confiance image : **{float(image_result.get('confidence', 0.0)):.0%}**")
        with res2:
            st.subheader("Association")
            st.metric("Compatibilité", f"{float(outfit_result.get('compatibility_score', 0.0)):.0%}")
            st.write(", ".join(outfit_result.get("recommended_product_types", [])) or "Aucune suggestion")
            st.write(f"Mode : **{outfit_result.get('mode', 'fallback')}**")
        with res3:
            st.subheader("Fit")
            st.metric("Prédiction", fit_result.get("fit_prediction", "unknown"))
            st.write(f"Confiance fit : **{float(fit_result.get('confidence', 0.0)):.0%}**")
            st.write(f"Risque : **{fit_result.get('risk_level', 'unknown')}**")

        st.subheader("Suggestions")
        for row in outfit_result.get("ranked_recommendations", []):
            st.write(
                f"**{row.get('product_type_v0')}** - ML {float(row.get('ml_score', 0.0)):.0%}, "
                f"couleur {float(row.get('color_harmony_score', 0.0)):.0%}, "
                f"cooccurrence {float(row.get('cooccurrence_score', 0.0)):.0%}"
            )
        if not outfit_result.get("ranked_recommendations"):
            st.write(", ".join(outfit_result.get("compatible_items", [])) or "Aucune suggestion")

        st.subheader("Conseil final")
        st.success(advice_result.get("advice", "Conseil indisponible."))
        for warning in advice_result.get("warnings", []):
            st.warning(warning)
        with st.expander("Détails techniques", expanded=False):
            st.json({"image": image_result, "fit": fit_result, "outfit": outfit_result, "advice": advice_result})

with outfit_tab:
    st.header("Images de la tenue")
    uploaded_outfit_files = st.file_uploader(
        "Ajoute deux images ou plus",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi_outfit_upload",
    )
    outfit_images = []
    if uploaded_outfit_files:
        preview_cols = st.columns(min(len(uploaded_outfit_files), 4))
        for index, uploaded in enumerate(uploaded_outfit_files):
            outfit_image = Image.open(uploaded).convert("RGB")
            outfit_images.append(outfit_image)
            with preview_cols[index % len(preview_cols)]:
                st.image(outfit_image, caption=f"Pièce {index + 1}", use_container_width=True)
    else:
        st.info("Ajoute plusieurs images pour obtenir un score global de tenue.")

    if st.button("Evaluer la tenue", type="primary"):
        outfit_evaluation = evaluate_outfit_images(outfit_images, context)
        st.metric("Score tenue", f"{float(outfit_evaluation.get('outfit_score', 0.0)):.0%}")
        st.write(f"Mode : **{outfit_evaluation.get('mode', 'fallback')}**")

        st.subheader("Pièces détectées")
        for item in outfit_evaluation.get("detected_items", []):
            st.write(
                f"**{item.get('product_type')}** - rôle {item.get('outfit_role')}, "
                f"couleur {item.get('color_label')}, confiance {float(item.get('confidence', 0.0)):.0%}"
            )

        st.subheader("Scores par paire")
        pair_scores = outfit_evaluation.get("pair_scores", [])
        if pair_scores:
            st.dataframe(pair_scores, use_container_width=True)
        else:
            st.write("Score pairwise ML indisponible tant qu'Outfit V2 n'est pas promu.")

        missing_roles = outfit_evaluation.get("missing_roles", [])
        if missing_roles:
            st.warning("Rôles manquants : " + ", ".join(missing_roles))
        suggestions = outfit_evaluation.get("suggested_associations", [])
        if suggestions:
            st.write("Associations proposées : " + ", ".join(suggestions))
        for warning in outfit_evaluation.get("warnings", []):
            st.warning(warning)
        with st.expander("Détails techniques", expanded=False):
            st.json(outfit_evaluation)
