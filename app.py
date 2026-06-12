import streamlit as st
import pandas as pd
import joblib
import os
import base64

# ==================================
# BASE_DIR (doit être en premier)
# ==================================

BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, "logo.png")

# ==================================
# Configuration
# ==================================

st.set_page_config(
    page_title="DrugSafe AI",
    page_icon=logo_path,
    layout="wide"
)

# ==================================
# CSS
# ==================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:900px;
}

h1{
    text-align:center;
}

</style>
""",
unsafe_allow_html=True)

# ==================================
# Chargement des fichiers
# ==================================

model = joblib.load(
    os.path.join(BASE_DIR, "drug_safety_model.pkl")
)

drug_encoder = joblib.load(
    os.path.join(BASE_DIR, "drug_encoder.pkl")
)

reaction_encoder = joblib.load(
    os.path.join(BASE_DIR, "reaction_encoder.pkl")
)

country_encoder = joblib.load(
    os.path.join(BASE_DIR, "country_encoder.pkl")
)

# ==================================
# Historique
# ==================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================
# Dictionnaires
# ==================================

sex_map = {
    "Homme": 1,
    "Femme": 2
}

qualification_map = {
    "Médecin": 1,
    "Pharmacien": 2,
    "Autre professionnel de santé": 3,
    "Avocat": 4,
    "Consommateur": 5
}

# ==================================
# Logo + Titre (centré via HTML)
# ==================================

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:0.5rem;">
            <img src="data:image/png;base64,{logo_b64}" width="110"/>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    "<h1 style='text-align:center; margin-top:0;'>DrugSafe AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='margin-top:-15px;color:gray;text-align:center;margin-bottom:1.5rem;'>
    Prédiction de la gravité des effets indésirables médicamenteux
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================
# Formulaire principal
# ==================================

with st.container(border=True):

    st.subheader("Informations du signalement")

    col1, col2 = st.columns(2)

    with col1:
        sexe = st.selectbox("Sexe", ["Homme", "Femme"])

    with col2:
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)

    qualification = st.selectbox(
        "Qualification du déclarant",
        [
            "Médecin",
            "Pharmacien",
            "Autre professionnel de santé",
            "Avocat",
            "Consommateur"
        ]
    )

    drug = st.selectbox(
        "Médicament",
        sorted(drug_encoder.classes_),
        index=None,
        placeholder="Rechercher un médicament..."
    )

    reaction = st.selectbox(
        "Effet indésirable",
        sorted(reaction_encoder.classes_),
        index=None,
        placeholder="Rechercher une réaction..."
    )

    country = st.selectbox(
        "Pays",
        sorted(country_encoder.classes_),
        index=None,
        placeholder="Rechercher un pays..."
    )

    predict_btn = st.button(
        "Lancer la prédiction",
        use_container_width=True,
        type="primary"
    )

# ==================================
# Prédiction
# ==================================

if predict_btn:

    if drug is None or reaction is None or country is None:
        st.warning("Veuillez compléter tous les champs.")

    else:

        drug_encoded          = drug_encoder.transform([drug])[0]
        reaction_encoded      = reaction_encoder.transform([reaction])[0]
        country_encoded       = country_encoder.transform([country])[0]
        sexe_encoded          = sex_map[sexe]
        qualification_encoded = qualification_map[qualification]

        X = pd.DataFrame(
            [[sexe_encoded, age, drug_encoded, reaction_encoded,
              country_encoded, qualification_encoded]],
            columns=["sex","age","drug","reaction","country","qualification"]
        )

        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max() * 100

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):

            st.subheader("Résultat de la prédiction")

            if prediction == 2:
                result_label = "Grave"
                st.markdown(
                    """
                    <div style='background-color:#fdecea;border-left:5px solid #e53935;
                    padding:12px 16px;border-radius:6px;color:#b71c1c;font-weight:600;'>
                    Effet indésirable grave
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                result_label = "Non grave"
                st.markdown(
                    """
                    <div style='background-color:#e8f5e9;border-left:5px solid #43a047;
                    padding:12px 16px;border-radius:6px;color:#1b5e20;font-weight:600;'>
                    Effet indésirable non grave
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.metric("Confiance du modèle", f"{confidence:.2f}%")
            st.progress(confidence / 100)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Résumé du cas")

            summary = pd.DataFrame({
                "Champ": ["Sexe","Âge","Médicament","Réaction","Pays","Qualification"],
                "Valeur": [sexe, age, drug, reaction, country, qualification]
            })

            st.dataframe(summary, hide_index=True, use_container_width=True)

        st.session_state.history.append({
            "Médicament": drug,
            "Réaction":   reaction,
            "Résultat":   result_label,
            "Confiance":  f"{confidence:.2f}%"
        })

# ==================================
# Historique
# ==================================

if st.session_state.history:

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Historique des prédictions", expanded=False):

        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.rerun()