"""
Interface interactive — Estimation de la valeur des biens immobiliers résidentiels au Sénégal
Mémoire M1, Dakar Institute of Technology.

Lancement local : streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "expat_dakar_listings_clean.csv"

FILE_KEY_TO_DISPLAY = {
    "linear_regression": "Régression linéaire",
    "random_forest": "Random Forest (optimisé)",
    "xgboost": "XGBoost (optimisé)",
    "mlp": "MLP (optimisé)",
}

st.set_page_config(
    page_title="Estimation immobilière Sénégal",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_metadata() -> dict:
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models() -> dict:
    metadata = load_metadata()
    models = {}
    for key, filename in metadata["model_files"].items():
        models[key] = joblib.load(MODELS_DIR / filename)
    return models


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    neighborhood_counts = df["neighborhood"].value_counts()
    frequent = neighborhood_counts[neighborhood_counts >= 5].index
    df["neighborhood_grouped"] = df["neighborhood"].where(df["neighborhood"].isin(frequent), other="Autre")
    return df


def predict_price(model, category: str, neighborhood: str, surface: float, bedrooms: int) -> float:
    row = pd.DataFrame([{
        "surface_m2": surface,
        "bedrooms": bedrooms,
        "category": category,
        "neighborhood_grouped": neighborhood,
    }])
    log_price = model.predict(row)[0]
    return float(np.expm1(log_price))


def format_fcfa(value: float) -> str:
    return f"{value:,.0f} FCFA".replace(",", " ")


metadata = load_metadata()
models = load_models()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisissez une section",
    ["Accueil", "Prédiction", "Exploration des données", "Comparaison des modèles"],
)

# ---------------------------------------------------------------------------
# ACCUEIL
# ---------------------------------------------------------------------------
if page == "Accueil":
    st.title("🏠 Estimation de la valeur des biens immobiliers résidentiels au Sénégal")
    st.markdown("""
    Projet Informatique — Master 1, Dakar Institute of Technology (Intelligence Artificielle).

    Cette application permet d'estimer le prix de vente d'un bien résidentiel (maison ou appartement)
    au Sénégal, à partir d'un modèle de machine learning entraîné sur des annonces réelles, et
    d'explorer les données et les performances des différents modèles comparés.
    """)

    col1, col2, col3 = st.columns(3)
    df = load_clean_data()
    col1.metric("Annonces utilisées", f"{len(df):,}".replace(",", " "))
    col2.metric("Quartiers couverts", df["neighborhood"].nunique())
    col3.metric("Meilleur modèle", FILE_KEY_TO_DISPLAY[metadata["best_model"]])

    st.subheader("Méthodologie en bref")
    st.markdown("""
    - **Source des données** : annonces de vente (maisons + appartements) publiées sur Expat-Dakar, collectées par scraping.
    - **Nettoyage** : suppression des prix non exploitables et des biens hors périmètre résidentiel unitaire (immeubles entiers).
    - **Modèles comparés** : Régression linéaire (référence), Random Forest, XGBoost et un réseau de neurones (MLP) —
      les trois derniers optimisés par recherche d'hyperparamètres avec validation croisée.
    - **Cible d'entraînement** : logarithme du prix (distribution très asymétrique), reconverti en FCFA pour l'évaluation.
    """)

    st.subheader("Performances des modèles (jeu de test)")
    results_df = pd.DataFrame(metadata["results"]).T
    results_df.index.name = "Modèle"
    st.dataframe(
        results_df.style.format({"RMSE_FCFA": "{:,.0f}", "MAE_FCFA": "{:,.0f}", "MAPE_%": "{:.1f}", "R2": "{:.3f}"}),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# PRÉDICTION
# ---------------------------------------------------------------------------
elif page == "Prédiction":
    st.title("Prédiction du prix d'un bien")

    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        st.subheader("Caractéristiques du bien")
        category = st.selectbox("Type de bien", metadata["categories"])
        neighborhood = st.selectbox("Quartier", metadata["neighborhoods"])
        surface_min, surface_max = metadata["surface_m2_range"]
        surface = st.slider("Surface (m²)", min_value=float(surface_min), max_value=float(surface_max), value=150.0, step=5.0)
        bedrooms_min, bedrooms_max = metadata["bedrooms_range"]
        bedrooms = st.slider("Nombre de chambres", min_value=int(bedrooms_min), max_value=int(bedrooms_max), value=3)

        model_key = st.selectbox(
            "Modèle utilisé pour la prédiction",
            options=list(FILE_KEY_TO_DISPLAY.keys()),
            format_func=lambda k: FILE_KEY_TO_DISPLAY[k],
            index=list(FILE_KEY_TO_DISPLAY.keys()).index(metadata["best_model"]),
        )
        predict_clicked = st.button("Estimer le prix", type="primary", width="stretch")

    with col_result:
        st.subheader("Résultat")
        if predict_clicked:
            model = models[model_key]
            predicted_price = predict_price(model, category, neighborhood, surface, bedrooms)

            model_display = FILE_KEY_TO_DISPLAY[model_key]
            model_metrics = metadata["results"][model_display]
            mae = model_metrics["MAE_FCFA"]
            mape = model_metrics["MAPE_%"]

            st.metric("Prix estimé", format_fcfa(predicted_price))
            st.caption(
                f"Fourchette approximative (± erreur absolue moyenne du modèle sur le jeu de test, "
                f"pas un intervalle de confiance statistique) : "
                f"{format_fcfa(max(0, predicted_price - mae))} — {format_fcfa(predicted_price + mae)}"
            )
            st.info(f"Modèle utilisé : **{model_display}** — MAPE observé sur le jeu de test : {mape:.1f}%")
        else:
            st.write("Renseignez les caractéristiques du bien et cliquez sur **Estimer le prix**.")

    st.divider()
    st.subheader("Comparer les prédictions des 4 modèles")
    if st.button("Comparer tous les modèles pour ces caractéristiques"):
        rows = []
        for key, display in FILE_KEY_TO_DISPLAY.items():
            price = predict_price(models[key], category, neighborhood, surface, bedrooms)
            rows.append({"Modèle": display, "Prix estimé (FCFA)": price})
        comparison_df = pd.DataFrame(rows)
        fig = px.bar(comparison_df, x="Modèle", y="Prix estimé (FCFA)", color="Modèle", text_auto=".2s")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(comparison_df.style.format({"Prix estimé (FCFA)": "{:,.0f}"}), width="stretch")

# ---------------------------------------------------------------------------
# EXPLORATION DES DONNÉES
# ---------------------------------------------------------------------------
elif page == "Exploration des données":
    st.title("Exploration des données")
    df = load_clean_data()

    with st.sidebar:
        st.subheader("Filtres")
        selected_categories = st.multiselect("Type de bien", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
        selected_neighborhoods = st.multiselect("Quartiers", sorted(df["neighborhood_grouped"].unique()))
        price_min, price_max = int(df["price_fcfa"].min()), int(df["price_fcfa"].max())
        price_range = st.slider("Fourchette de prix (FCFA)", price_min, price_max, (price_min, price_max))

    filtered = df[df["category"].isin(selected_categories)]
    if selected_neighborhoods:
        filtered = filtered[filtered["neighborhood_grouped"].isin(selected_neighborhoods)]
    filtered = filtered[filtered["price_fcfa"].between(*price_range)]

    st.caption(f"{len(filtered)} annonces correspondant aux filtres (sur {len(df)} au total)")

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(filtered, x="price_fcfa", nbins=40, title="Distribution des prix")
        st.plotly_chart(fig_hist, width="stretch")
    with col2:
        fig_scatter = px.scatter(
            filtered, x="surface_m2", y="price_fcfa", color="category",
            hover_data=["neighborhood"], title="Prix en fonction de la surface",
        )
        st.plotly_chart(fig_scatter, width="stretch")

    median_by_neighborhood = (
        filtered.groupby("neighborhood")["price_fcfa"]
        .agg(["median", "count"])
        .query("count >= 3")
        .sort_values("median", ascending=False)
        .head(15)
        .reset_index()
    )
    fig_bar = px.bar(
        median_by_neighborhood, x="median", y="neighborhood", orientation="h",
        title="Prix médian par quartier (quartiers avec ≥ 3 annonces filtrées)",
        labels={"median": "Prix médian (FCFA)", "neighborhood": "Quartier"},
    )
    fig_bar.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Données filtrées")
    st.dataframe(
        filtered[["title", "category", "neighborhood", "city_region", "price_fcfa", "surface_m2", "bedrooms"]],
        width="stretch",
    )

# ---------------------------------------------------------------------------
# COMPARAISON DES MODÈLES
# ---------------------------------------------------------------------------
elif page == "Comparaison des modèles":
    st.title("Comparaison des modèles")

    results_df = pd.DataFrame(metadata["results"]).T
    results_df.index.name = "Modèle"
    results_df = results_df.reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(results_df, x="Modèle", y="RMSE_FCFA", title="RMSE (FCFA) — plus bas = meilleur", color="Modèle")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(results_df, x="Modèle", y="MAPE_%", title="MAPE (%) — plus bas = meilleur", color="Modèle")
        st.plotly_chart(fig, width="stretch")
    with col3:
        fig = px.bar(results_df, x="Modèle", y="R2", title="R² — plus haut = meilleur", color="Modèle")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Tableau détaillé")
    st.dataframe(
        results_df.style.format({"RMSE_FCFA": "{:,.0f}", "MAE_FCFA": "{:,.0f}", "MAPE_%": "{:.1f}", "R2": "{:.3f}"}),
        width="stretch",
    )

    st.markdown("""
    **Lecture des résultats** : le Random Forest optimisé obtient les meilleures performances sur ce jeu de données
    (~600 annonces), y compris face au réseau de neurones (MLP). Sur un volume de données de cet ordre de grandeur,
    les méthodes d'ensemble à base d'arbres généralisent généralement mieux qu'un réseau de neurones, qui a besoin
    de davantage de données pour exprimer son avantage — un constat cohérent avec la littérature en apprentissage
    automatique sur données tabulaires de petite/moyenne taille.
    """)
