"""
tests/test_retriever.py — Tests unitaires pour src/retriever.py.
─────────────────────────────────────────────────────────────────
Lancement : pytest tests/test_retriever.py -v

Ces tests utilisent une ChromaDB temporaire en mémoire
pour ne pas dépendre de la base de prod.
"""

import pytest
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.retriever import search, search_with_scores


@pytest.fixture(scope="module")
def test_vectorstore():
    """
    Crée une base ChromaDB en mémoire avec des documents de test.
    Partagée par tous les tests du module (scope=module).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    docs = [
        Document(
            page_content="Le LOD 350 est requis pour les équipements MEP en phase EXE.",
            metadata={"source": "bep_projet_a.pdf", "page": 12},
        ),
        Document(
            page_content="Convention de nommage IFC : [Projet]-[Discipline]-[Phase]-[Zone].ifc",
            metadata={"source": "charte_bim.pdf", "page": 5},
        ),
        Document(
            page_content="En cas de clash critique, le BIM Manager doit être notifié sous 24h.",
            metadata={"source": "procedure_coordination.pdf", "page": 3},
        ),
        Document(
            page_content="Les logiciels autorisés pour la production IFC sont : Revit, Archicad, Tekla.",
            metadata={"source": "bep_projet_a.pdf", "page": 8},
        ),
        Document(
            page_content="Les réunions de coordination BIM ont lieu chaque lundi à 9h.",
            metadata={"source": "organisation_projet.pdf", "page": 1},
        ),
    ]

    # ChromaDB en mémoire (pas de persist_directory)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    return vectorstore


class TestSearch:

    def test_retourne_resultats(self, test_vectorstore):
        """La recherche doit retourner des résultats."""
        results = search(test_vectorstore, "LOD MEP", top_k=3)
        assert len(results) > 0

    def test_top_k_respecte(self, test_vectorstore):
        """Le nombre de résultats ne doit pas dépasser top_k."""
        results = search(test_vectorstore, "BIM", top_k=2)
        assert len(results) <= 2

    def test_resultats_sont_des_documents(self, test_vectorstore):
        """Les résultats doivent être des instances Document."""
        results = search(test_vectorstore, "nommage IFC", top_k=3)
        for doc in results:
            assert isinstance(doc, Document)

    def test_documents_ont_metadonnees(self, test_vectorstore):
        """Chaque document doit avoir source et page dans ses métadonnées."""
        results = search(test_vectorstore, "clash critique", top_k=1)
        assert len(results) > 0
        doc = results[0]
        assert "source" in doc.metadata
        assert "page" in doc.metadata

    def test_pertinence_semantique(self, test_vectorstore):
        """Le document le plus pertinent doit correspondre à la requête."""
        results = search(test_vectorstore, "procédure clash coordination", top_k=1)
        assert len(results) > 0
        # Le doc sur les clashs doit être en tête
        assert "clash" in results[0].page_content.lower()

    def test_requete_vide(self, test_vectorstore):
        """Une requête vide ne doit pas lever d'exception."""
        results = search(test_vectorstore, "", top_k=3)
        assert isinstance(results, list)


class TestSearchWithScores:

    def test_retourne_tuples_doc_score(self, test_vectorstore):
        """Doit retourner des tuples (Document, float)."""
        results = search_with_scores(test_vectorstore, "LOD MEP", top_k=3)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_scores_positifs(self, test_vectorstore):
        """Les scores de similarité doivent être positifs."""
        results = search_with_scores(test_vectorstore, "nommage IFC", top_k=3)
        for _, score in results:
            assert score >= 0
