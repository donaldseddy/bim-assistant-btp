# ─────────────────────────────────────────────────────────────────
# Dockerfile — BIM Assistant BTP
# Build  : docker build -t bim-assistant-btp .
# Run    : docker-compose up
# ─────────────────────────────────────────────────────────────────

FROM python:3.13-slim

# Métadonnées
LABEL maintainer="equipe-bim@entreprise.fr"
LABEL description="BIM Assistant BTP — Chatbot RAG Mistral 7B"

# Répertoire de travail dans le conteneur
WORKDIR /app

# ── Dépendances système ───────────────────────────────────────────
# Nécessaire pour ChromaDB et sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ────────────────────────────────────────────
# Copier requirements en premier pour profiter du cache Docker
# (si requirements.txt n'a pas changé, cette couche est réutilisée)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────
COPY . .

# ── Créer les dossiers de données ─────────────────────────────────
RUN mkdir -p data/raw data/chroma_db logs

# ── Port Streamlit ────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Commande de démarrage ─────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
