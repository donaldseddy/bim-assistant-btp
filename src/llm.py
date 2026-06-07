"""
src/llm.py — Couche LLM : connexion HuggingFace + construction du prompt RAG.
─────────────────────────────────────────────────────────────────────────────
Responsabilité unique :
  - Charger et exposer le modèle Mistral via HuggingFaceEndpoint
  - Construire le prompt final (system_prompt + contexte RAG + question)
  - Appeler le modèle et retourner la réponse

Ce module ne sait rien de Streamlit ni de ChromaDB → découplage propre.
"""

import os
import logging
from typing import List

import yaml
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document

load_dotenv()
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Charge la configuration depuis config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_llm(config: dict) -> HuggingFaceEndpoint:
    """
    Instancie le modèle LLM HuggingFace.

    Appelé une seule fois au démarrage de l'app (st.cache_resource).
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN manquant. "
            "Vérifier le fichier .env à la racine du projet."
        )

    llm_cfg = config["llm"]
    logger.info(f"Chargement du modèle : {llm_cfg['repo_id']}")

    return HuggingFaceEndpoint(
        repo_id=llm_cfg["repo_id"],
        max_new_tokens=llm_cfg["max_new_tokens"],
        temperature=llm_cfg["temperature"],
        task=llm_cfg["task"],
        huggingfacehub_api_token=token,
    )


def build_prompt(
    question: str,
    documents: List[Document],
    system_prompt: str,
) -> str:
    """
    Construit le prompt final envoyé à Mistral.

    Structure :
      [SYSTEM PROMPT]
      [CONTEXTE DOCUMENTAIRE — passages extraits de ChromaDB]
      [QUESTION UTILISATEUR]

    Args:
        question: La question posée par l'utilisateur.
        documents: Liste de documents récupérés par le retriever.
        system_prompt: Instructions système depuis config.yaml.

    Returns:
        Le prompt complet sous forme de chaîne de caractères.
    """
    if documents:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Document inconnu")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"--- Source {i} : {source} (page {page}) ---\n{doc.page_content}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "Aucun document pertinent trouvé dans la base."

    prompt = f"""{system_prompt}

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION : {question}

RÉPONSE :"""

    logger.debug(f"Prompt construit ({len(prompt)} caractères)")
    return prompt


def get_response(
    llm: HuggingFaceEndpoint,
    question: str,
    documents: List[Document],
    system_prompt: str,
) -> str:
    """
    Appelle le LLM et retourne la réponse.

    Args:
        llm: Instance HuggingFaceEndpoint.
        question: Question de l'utilisateur.
        documents: Documents récupérés par le retriever.
        system_prompt: Instructions système.

    Returns:
        Réponse générée par Mistral.
    """
    prompt = build_prompt(question, documents, system_prompt)

    try:
        response = llm.invoke(prompt)
        logger.info(f"Réponse générée ({len(response)} caractères)")
        return response.strip()
    except Exception as e:
        logger.error(f"Erreur LLM : {e}")
        raise
