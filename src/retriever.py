"""
src/retriever.py — Couche de recherche vectorielle dans ChromaDB.
─────────────────────────────────────────────────────────────────
Responsabilité unique :
  - Charger la base ChromaDB existante (créée par ingest.py)
  - Exposer une fonction de recherche top-k par similarité
  - Retourner les documents avec leurs métadonnées (source, page)

Ce module ne sait rien de Streamlit ni du LLM → découplage propre.
Prérequis : avoir lancé `python src/ingest.py` au moins une fois.
"""

import logging
from typing import List, Tuple

import yaml
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_retriever(config: dict) -> Chroma:
    """
    Charge la base ChromaDB persistante.

    Appelé une seule fois au démarrage (st.cache_resource dans app.py).

    Args:
        config: Configuration depuis config.yaml.

    Returns:
        Instance Chroma prête à être requêtée.

    Raises:
        FileNotFoundError: Si la base ChromaDB n'existe pas encore.
    """
    chroma_cfg = config["chroma"]
    retriever_cfg = config["retriever"]

    persist_dir = chroma_cfg["persist_directory"]

    import os
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Base ChromaDB introuvable : {persist_dir}\n"
            "Lance d'abord : python src/ingest.py --source data/raw/"
        )

    logger.info(f"Chargement ChromaDB depuis : {persist_dir}")

    embeddings = HuggingFaceEmbeddings(
        model_name=retriever_cfg["embedding_model"],
        model_kwargs={"device": "cpu"},
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=chroma_cfg["collection_name"],
    )

    count = vectorstore._collection.count()
    logger.info(f"ChromaDB chargée — {count} chunks disponibles")

    return vectorstore


def search(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
) -> List[Document]:
    """
    Recherche les passages les plus pertinents pour une question.

    Args:
        vectorstore: Instance Chroma chargée.
        query: Question posée par l'utilisateur.
        top_k: Nombre de passages à récupérer.

    Returns:
        Liste de documents avec contenu et métadonnées (source, page).
    """
    logger.info(f"Recherche — query: '{query[:60]}...' | top_k={top_k}")

    results = vectorstore.similarity_search(query, k=top_k)

    logger.info(f"  → {len(results)} passage(s) récupéré(s)")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        logger.debug(f"  [{i}] {source} p.{page} — {doc.page_content[:80]}...")

    return results


def search_with_scores(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Recherche avec scores de similarité (pour debug et expérimentations).

    Returns:
        Liste de tuples (document, score_cosinus).
    """
    return vectorstore.similarity_search_with_score(query, k=top_k)
