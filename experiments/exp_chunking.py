"""
experiments/exp_chunking.py — Benchmark des paramètres de chunking.
────────────────────────────────────────────────────────────────────
Expérience 2 du README : trouver le chunk_size optimal pour les
documents BIM (BEP, DTU, fiches techniques).

Usage :
  python experiments/exp_chunking.py --source data/raw/sample.pdf

Résultats attendus (documentés dans le README) :
  chunk_size=500  → trop fragmenté, perd le contexte des tableaux
  chunk_size=800  → meilleur équilibre (retenu)
  chunk_size=1200 → trop long, dépasse la fenêtre contexte Mistral 7B
"""

import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def benchmark_chunking(source_path: str):
    print(f"\n📄 Source : {source_path}\n")

    loader = PyPDFLoader(source_path)
    pages = loader.load()
    print(f"Pages chargées : {len(pages)}\n")

    configurations = [
        {"chunk_size": 500,  "chunk_overlap": 50},
        {"chunk_size": 800,  "chunk_overlap": 100},
        {"chunk_size": 1200, "chunk_overlap": 150},
    ]

    print(f"{'chunk_size':<12} {'chunk_overlap':<15} {'nb_chunks':<12} {'avg_chars':<12} {'observation'}")
    print("-" * 80)

    for cfg in configurations:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)

        # Heuristique d'observation
        if avg_len < 400:
            obs = "⚠️  Trop fragmenté"
        elif avg_len > 900:
            obs = "⚠️  Risque dépassement contexte"
        else:
            obs = "✅  Équilibre optimal"

        print(f"{cfg['chunk_size']:<12} {cfg['chunk_overlap']:<15} {len(chunks):<12} {avg_len:<12.0f} {obs}")

    print("\n💡 Recommandation : chunk_size=800, chunk_overlap=100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Chemin vers un PDF source")
    args = parser.parse_args()
    benchmark_chunking(args.source)
