"""
tests/test_llm.py — Tests unitaires pour src/llm.py.
─────────────────────────────────────────────────────
Lancement : pytest tests/test_llm.py -v

Ces tests vérifient la logique de construction du prompt
sans appeler l'API HuggingFace (pas de coût, pas de réseau).
"""

import pytest
from langchain.schema import Document
from src.llm import build_prompt, load_config


SYSTEM_PROMPT = "Tu es un assistant BIM expert."


def make_doc(content: str, source: str = "bep_projet_a.pdf", page: int = 1) -> Document:
    """Helper : crée un Document de test."""
    return Document(
        page_content=content,
        metadata={"source": source, "page": page},
    )


class TestBuildPrompt:

    def test_prompt_contient_question(self):
        """La question doit apparaître dans le prompt."""
        question = "Quel est le LOD requis pour les éléments MEP ?"
        docs = [make_doc("LOD 350 requis pour les équipements MEP en phase EXE.")]
        prompt = build_prompt(question, docs, SYSTEM_PROMPT)
        assert question in prompt

    def test_prompt_contient_system_prompt(self):
        """Le system prompt doit apparaître dans le prompt."""
        prompt = build_prompt("Question test", [], SYSTEM_PROMPT)
        assert SYSTEM_PROMPT in prompt

    def test_prompt_contient_contenu_document(self):
        """Le contenu des documents doit apparaître dans le prompt."""
        contenu = "Convention de nommage : [Projet]-[Discipline]-[Phase].ifc"
        docs = [make_doc(contenu)]
        prompt = build_prompt("Comment nommer un IFC ?", docs, SYSTEM_PROMPT)
        assert contenu in prompt

    def test_prompt_cite_source_document(self):
        """Le nom du fichier source doit apparaître dans le prompt."""
        docs = [make_doc("Contenu", source="charte_bim_v2.pdf", page=5)]
        prompt = build_prompt("Question", docs, SYSTEM_PROMPT)
        assert "charte_bim_v2.pdf" in prompt

    def test_prompt_sans_documents(self):
        """Sans documents, le prompt doit indiquer l'absence de contexte."""
        prompt = build_prompt("Question", [], SYSTEM_PROMPT)
        assert "Aucun document pertinent" in prompt

    def test_prompt_plusieurs_documents(self):
        """Plusieurs documents doivent tous apparaître dans le prompt."""
        docs = [
            make_doc("Passage 1", source="doc1.pdf"),
            make_doc("Passage 2", source="doc2.pdf"),
            make_doc("Passage 3", source="doc3.pdf"),
        ]
        prompt = build_prompt("Question", docs, SYSTEM_PROMPT)
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt
        assert "doc3.pdf" in prompt

    def test_prompt_non_vide(self):
        """Le prompt ne doit jamais être vide."""
        prompt = build_prompt("", [], "")
        assert len(prompt) > 0


class TestLoadConfig:

    def test_config_charge_correctement(self):
        """La config doit se charger sans erreur."""
        config = load_config("config.yaml")
        assert "llm" in config
        assert "retriever" in config
        assert "chroma" in config
        assert "system_prompt" in config

    def test_config_llm_contient_repo_id(self):
        config = load_config("config.yaml")
        assert "repo_id" in config["llm"]
        assert "Mistral" in config["llm"]["repo_id"]

    def test_config_retriever_valeurs_valides(self):
        config = load_config("config.yaml")
        assert config["retriever"]["chunk_size"] > 0
        assert config["retriever"]["chunk_overlap"] >= 0
        assert config["retriever"]["top_k"] > 0
