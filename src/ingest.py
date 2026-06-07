"""
src/ingest.py — Pipeline d'ingestion documentaire vers ChromaDB.
─────────────────────────────────────────────────────────────────
Responsabilité unique :
  - Lire les documents depuis data/raw/ (PDF, DOCX, TXT)
  - Les découper en chunks (RecursiveCharacterTextSplitter)
  - Générer les embeddings (CamemBERT)
  - Persister dans ChromaDB

Usage CLI :
  python src/ingest.py --source data/raw/
  python src/ingest.py --source data/raw/mon_bep.pdf

À relancer chaque fois que de nouveaux documents sont ajoutés dans data/raw/.
Ne pas relancer l'application Streamlit pour ça.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import yaml
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Extensions supportées et leur loader associé
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_documents(source_path: str) -> List[Document]:
    """
    Charge tous les documents depuis un fichier ou un dossier.

    Args:
        source_path: Chemin vers un fichier ou un répertoire.

    Returns:
        Liste de documents LangChain avec métadonnées (source, page).
    """
    path = Path(source_path)
    documents = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [f for f in path.rglob("*") if f.suffix.lower() in LOADERS]
    else:
        raise FileNotFoundError(f"Chemin introuvable : {source_path}")

    if not files:
        logger.warning(f"Aucun fichier supporté trouvé dans {source_path}")
        return documents

    for file in files:
        loader_class = LOADERS.get(file.suffix.lower())
        if not loader_class:
            logger.debug(f"Format non supporté, ignoré : {file.name}")
            continue

        try:
            logger.info(f"Chargement : {file.name}")
            loader = loader_class(str(file))
            docs = loader.load()

            # Enrichir les métadonnées avec le nom de fichier
            for doc in docs:
                doc.metadata["source"] = file.name

            documents.extend(docs)
            logger.info(f"  → {len(docs)} page(s) chargée(s)")
        except Exception as e:
            logger.error(f"  ❌ Erreur sur {file.name} : {e}")

    logger.info(f"Total : {len(documents)} pages chargées depuis {len(files)} fichier(s)")
    return documents


def split_documents(documents: List[Document], config: dict) -> List[Document]:
    """
    Découpe les documents en chunks selon la config.

    Args:
        documents: Liste brute de documents.
        config: Configuration depuis config.yaml.

    Returns:
        Liste de chunks prêts pour l'embedding.
    """
    retriever_cfg = config["retriever"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=retriever_cfg["chunk_size"],
        chunk_overlap=retriever_cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Découpage : {len(documents)} pages → {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: List[Document], config: dict) -> Chroma:
    """
    Génère les embeddings et persiste dans ChromaDB.

    Args:
        chunks: Liste de chunks à indexer.
        config: Configuration depuis config.yaml.

    Returns:
        Instance ChromaDB prête à être requêtée.
    """
    retriever_cfg = config["retriever"]
    chroma_cfg = config["chroma"]

    logger.info(f"Chargement du modèle d'embeddings : {retriever_cfg['embedding_model']}")
    embeddings = HuggingFaceEmbeddings(
        model_name=retriever_cfg["embedding_model"],
        model_kwargs={"device": "cpu"},
    )

    logger.info(f"Indexation de {len(chunks)} chunks dans ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_cfg["persist_directory"],
        collection_name=chroma_cfg["collection_name"],
    )

    logger.info(f"✅ {len(chunks)} chunks indexés dans {chroma_cfg['persist_directory']}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion de documents BIM dans ChromaDB"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/",
        help="Chemin vers un fichier ou un dossier source (défaut: data/raw/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers la configuration (défaut: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Créer le dossier de persistance si nécessaire
    os.makedirs(config["chroma"]["persist_directory"], exist_ok=True)

    documents = load_documents(args.source)
    if not documents:
        logger.error("Aucun document chargé. Vérifier le dossier data/raw/")
        return

    chunks = split_documents(documents, config)
    build_vectorstore(chunks, config)

    logger.info("✅ Ingestion terminée. Tu peux lancer : streamlit run app.py")


if __name__ == "__main__":
    main()
