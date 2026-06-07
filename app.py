"""
app.py — Point d'entrée principal de l'application Streamlit.
─────────────────────────────────────────────────────────────────
Lancement : streamlit run app.py

Ce fichier est volontairement court.
Toute la logique est dans src/ → app.py orchestre uniquement.

Architecture de la page :
  1. Chargement config + modèles (mis en cache par Streamlit)
  2. Sidebar (paramètres utilisateur)
  3. Historique de conversation
  4. Saisie utilisateur → pipeline RAG → affichage réponse + sources
"""

import logging
import os
import sys

import streamlit as st

# ─── Configuration page ───────────────────────────────────────────
st.set_page_config(
    page_title="BIM Assistant BTP",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports internes ─────────────────────────────────────────────
from src.llm import load_config, get_llm, get_response
from src.retriever import get_retriever, search
from src.ui import (
    render_sidebar,
    render_chat_history,
    render_sources,
    render_empty_state,
    render_error,
    render_warning,
)

# ─── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Chargement des ressources (mis en cache) ─────────────────────

@st.cache_resource
def load_resources():
    """
    Charge config, LLM et ChromaDB une seule fois au démarrage.
    st.cache_resource évite de recharger à chaque interaction.
    """
    config = load_config()
    llm = get_llm(config)

    try:
        vectorstore = get_retriever(config)
    except FileNotFoundError as e:
        # ChromaDB pas encore initialisée → mode dégradé sans RAG
        logger.warning(str(e))
        vectorstore = None

    return config, llm, vectorstore


# ─── Main ─────────────────────────────────────────────────────────

def main():
    # Chargement des ressources
    try:
        config, llm, vectorstore = load_resources()
    except EnvironmentError as e:
        render_error(str(e))
        st.stop()

    # Avertissement si pas de base vectorielle
    if vectorstore is None:
        render_warning(
            "Base documentaire non initialisée. "
            "Lance : `python src/ingest.py --source data/raw/` "
            "pour activer le RAG. En attendant, l'assistant répond sans contexte."
        )

    # Sidebar
    options = render_sidebar()

    # Initialisation de l'historique de session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Titre principal
    st.title("🏗️ BIM Assistant BTP")
    st.caption("Assistant RAG pour les équipes BIM — Modélisation · MEP · Travaux · Coordination")
    st.divider()

    # Affichage de l'état vide ou de l'historique
    if not st.session_state.messages:
        render_empty_state()
    else:
        render_chat_history(st.session_state.messages)

    # Zone de saisie
    question = st.chat_input("Posez votre question BIM...")

    if question:
        # Afficher la question de l'utilisateur
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Pipeline RAG
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):

                # 1. Récupérer les passages pertinents
                documents = []
                if vectorstore is not None:
                    documents = search(
                        vectorstore=vectorstore,
                        query=question,
                        top_k=options["top_k"],
                    )

                # 2. Générer la réponse
                try:
                    response = get_response(
                        llm=llm,
                        question=question,
                        documents=documents,
                        system_prompt=config["system_prompt"],
                    )
                except Exception as e:
                    render_error(f"Erreur lors de la génération : {e}")
                    st.stop()

            # 3. Afficher la réponse
            st.markdown(response)

            # 4. Afficher les sources si activé
            if options["show_sources"] and documents:
                render_sources(documents)

        # Sauvegarder la réponse dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Bouton reset conversation
    if st.session_state.messages:
        if st.button("🗑️ Nouvelle conversation", use_container_width=False):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
