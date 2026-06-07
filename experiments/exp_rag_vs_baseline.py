"""
experiments/exp_rag_vs_baseline.py — RAG vs Baseline (sans RAG).
────────────────────────────────────────────────────────────────────
Expérience 4 du README : mesurer l'apport du RAG sur des questions
représentatives du périmètre BIM.

Prérequis :
  - HUGGINGFACEHUB_API_TOKEN dans .env
  - Base ChromaDB initialisée : python src/ingest.py --source data/raw/

Usage :
  python experiments/exp_rag_vs_baseline.py
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

from src.llm import load_config, build_prompt
from src.retriever import get_retriever, search

load_dotenv()

QUESTIONS = [
    "Quelle est la convention de nommage des familles Revit sur ce projet ?",
    "Quel est le LOD requis pour les éléments de charpente en phase EXE ?",
    "Qui est responsable de la coordination MEP selon notre BEP ?",
    "Quels sont les logiciels autorisés pour la production IFC ?",
    "Quelle est la procédure en cas de clash critique détecté en coordination ?",
]


def run():
    config = load_config()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceEndpoint(
        repo_id=config["llm"]["repo_id"],
        max_new_tokens=300,
        temperature=config["llm"]["temperature"],
        task=config["llm"]["task"],
        huggingfacehub_api_token=token,
    )

    try:
        vectorstore = get_retriever(config)
        rag_available = True
    except FileNotFoundError:
        print("⚠️  ChromaDB non initialisée — mode baseline uniquement.\n")
        rag_available = False

    system_prompt = config["system_prompt"]

    print("=" * 80)
    print("EXPÉRIENCE 4 — RAG vs Baseline")
    print("=" * 80)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] {question}")
        print("-" * 60)

        # ── Baseline (sans RAG) ──────────────────────────────────
        prompt_baseline = build_prompt(question, [], system_prompt)
        response_baseline = llm.invoke(prompt_baseline).strip()
        print(f"❌ SANS RAG :\n{response_baseline[:300]}...")

        # ── Avec RAG ─────────────────────────────────────────────
        if rag_available:
            documents = search(vectorstore, question, top_k=config["retriever"]["top_k"])
            prompt_rag = build_prompt(question, documents, system_prompt)
            response_rag = llm.invoke(prompt_rag).strip()
            sources = [d.metadata.get("source", "?") for d in documents]
            print(f"\n✅ AVEC RAG :\n{response_rag[:300]}...")
            print(f"   📎 Sources : {', '.join(sources)}")
        else:
            print("✅ AVEC RAG : non disponible (lancer ingest.py d'abord)")

    print("\n" + "=" * 80)
    print("Expérience terminée.")


if __name__ == "__main__":
    run()
