"""
experiments/exp_embeddings.py — Comparaison modèles d'embeddings.
────────────────────────────────────────────────────────────────────
Expérience 3 du README : trouver le modèle d'embeddings le plus
pertinent pour des documents BIM en français.

Usage :
  python experiments/exp_embeddings.py

Modèles testés :
  - all-MiniLM-L6-v2            : léger, multilingue
  - paraphrase-multilingual-mpnet: FR/EN équilibré
  - sentence-camembert-base      : français natif (retenu)
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "dangvantuan/sentence-camembert-base",
]

# Passage extrait d'un BEP (anonymisé)
REFERENCE = (
    "Les fichiers IFC doivent être nommés selon la convention : "
    "[Projet]-[Discipline]-[Phase]-[Zone].ifc"
)

QUERIES = [
    ("Proche",           "Comment nommer un fichier IFC ?"),
    ("Sémantique",       "Quelle est la convention de nommage des maquettes ?"),
    ("Hors sujet",       "Quel logiciel utiliser pour ouvrir un fichier IFC ?"),
]


def run():
    print("\n📊 Benchmark modèles d'embeddings — documents BIM FR\n")
    print(f"Référence : {REFERENCE[:80]}...\n")

    header = f"{'Modèle':<50} {'Proche':>10} {'Sémantique':>12} {'Hors sujet':>12}"
    print(header)
    print("-" * 90)

    best_model = None
    best_score = -1

    for model_name in MODELS:
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )
        ref_emb = embedder.embed_query(REFERENCE)

        scores = {}
        for label, query in QUERIES:
            q_emb = embedder.embed_query(query)
            score = cosine_similarity([ref_emb], [q_emb])[0][0]
            scores[label] = score

        # Score composite : proche + sémantique - hors_sujet
        composite = scores["Proche"] + scores["Sémantique"] - scores["Hors sujet"]

        short_name = model_name.split("/")[-1]
        print(
            f"{short_name:<50} "
            f"{scores['Proche']:>10.3f} "
            f"{scores['Sémantique']:>12.3f} "
            f"{scores['Hors sujet']:>12.3f}"
        )

        if composite > best_score:
            best_score = composite
            best_model = model_name

    print(f"\n✅ Modèle recommandé : {best_model}")
    print("   → Meilleure discrimination entre requêtes pertinentes et hors sujet.")


if __name__ == "__main__":
    run()
