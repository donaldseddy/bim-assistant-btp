"""
src/ui.py — Composants Streamlit réutilisables.
─────────────────────────────────────────────────────────────────
Responsabilité unique :
  - Centraliser le rendu de l'interface (sidebar, messages, sources)
  - Ne contient aucune logique métier (pas d'appel LLM, pas de ChromaDB)
  - Facilite la maintenance : modifier l'UI sans toucher app.py

Pattern : chaque fonction reçoit des données, affiche, retourne rien.
"""

import streamlit as st
from langchain.schema import Document
from typing import List


def render_sidebar() -> dict:
    """
    Affiche la sidebar avec les paramètres configurables par l'utilisateur.

    Returns:
        Dictionnaire des options choisies par l'utilisateur.
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/building.png",
            width=60,
        )
        st.title("BIM Assistant BTP")
        st.caption("Propulsé par Mistral 7B · ChromaDB")

        st.divider()

        st.subheader("⚙️ Paramètres")

        top_k = st.slider(
            "Passages récupérés (top-k)",
            min_value=1,
            max_value=10,
            value=5,
            help="Nombre de passages documentaires injectés dans le contexte.",
        )

        show_sources = st.toggle(
            "Afficher les sources",
            value=True,
            help="Affiche les documents utilisés pour générer la réponse.",
        )

        st.divider()

        st.subheader("📂 Équipe")
        equipe = st.selectbox(
            "Filtrer par équipe",
            options=["Toutes", "Modélisation", "MEP", "Travaux", "BIM Management"],
            help="Filtre les documents selon l'équipe concernée (Phase 4).",
        )

        st.divider()
        st.caption("🔒 Données confidentielles — usage interne uniquement.")

    return {
        "top_k": top_k,
        "show_sources": show_sources,
        "equipe": equipe,
    }


def render_chat_history(messages: List[dict]) -> None:
    """
    Affiche l'historique de conversation.

    Args:
        messages: Liste de dict {"role": "user"|"assistant", "content": str}
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sources(documents: List[Document]) -> None:
    """
    Affiche les sources documentaires utilisées pour la réponse.

    Args:
        documents: Documents récupérés par le retriever.
    """
    if not documents:
        return

    with st.expander(f"📎 Sources utilisées ({len(documents)} passage(s))", expanded=False):
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Document inconnu")
            page = doc.metadata.get("page", "?")

            st.markdown(f"**[{i}] {source}** — page {page}")
            st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

            if i < len(documents):
                st.divider()


def render_empty_state() -> None:
    """Affiche l'état vide quand aucune conversation n'a démarré."""
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0; color: #888;">
            <h3>🏗️ Bienvenue sur BIM Assistant BTP</h3>
            <p>Posez une question sur vos documents BIM internes :</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Conventions de nommage du projet X ?</li>
                <li>LOD requis pour la charpente en phase EXE ?</li>
                <li>Procédure en cas de clash critique ?</li>
                <li>Logiciels IFC autorisés selon notre BEP ?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error(message: str) -> None:
    """Affiche un message d'erreur formaté."""
    st.error(f"❌ {message}", icon="🚨")


def render_warning(message: str) -> None:
    """Affiche un avertissement formaté."""
    st.warning(f"⚠️ {message}")
