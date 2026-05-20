# 🏗️ BIM Assistant BTP

> Chatbot interne RAG pour les équipes BIM — modélisation, coordination, travaux, MEP.  
> Propulsé par **Mistral 7B** via Hugging Face — 100% open source, zéro coût LLM.

---

##  Objectif

Mettre à disposition des équipes BIM un assistant conversationnel capable de répondre aux questions métier en s'appuyant sur la **base documentaire interne** de l'entreprise : BEP, DTU, standards de modélisation, fiches techniques, comptes-rendus de réunion.

L'assistant évolue progressivement vers une intégration CDE (BIM 360 / ACC) et une lecture directe des maquettes IFC.

---


### Phase 0 — Fondations (Semaine 1)
> *"Don't build on sand."*

- [ ] Initialiser le dépôt Git avec cette structure de projet
- [ ] Créer l'environnement virtuel Python (`venv` ou `conda`)
- [ ] Configurer `.env` pour les secrets (`HUGGINGFACEHUB_API_TOKEN`)
- [ ] Mettre en place `.gitignore` (`venv/`, `.env`, `__pycache__/`, `data/chroma_db/`, `data/raw/`)
- [ ] Valider que le modèle Mistral répond (script de test minimal `test_llm.py`)

**Livrable :** `python test_llm.py` retourne une réponse de Mistral 7B. ✅

---

### Phase 1 — MVP Chatbot simple (Semaine 2)
> *"Make it work."*

- [ ] Interface Streamlit basique : champ texte + affichage réponse en streaming
- [ ] Appel `HuggingFaceEndpoint` avec `Mistral-7B-Instruct-v0.3` sans RAG
- [ ] System prompt métier BIM injecté à chaque session
- [ ] Historique de conversation dans `st.session_state`
- [ ] Gestion des erreurs (timeout HF, token expiré, modèle indisponible)

**Livrable :** Un chatbot BIM qui répond de façon générale, déployable localement. ✅

---

### Phase 2 — RAG sur documents internes (Semaines 3–4)
> *"Make it smart."*

- [ ] Pipeline d'ingestion documentaire (`data/raw/` → chunks → embeddings)
  - Formats supportés : PDF, DOCX, TXT
  - Chunking avec LangChain `RecursiveCharacterTextSplitter`
  - Embeddings avec `sentence-transformers/all-MiniLM-L6-v2` (local, gratuit)
- [ ] Base vectorielle locale avec **ChromaDB** (persistance sur disque)
- [ ] Retriever : top-k passages pertinents injectés dans le prompt Mistral
- [ ] Affichage des **sources citées** dans l'interface Streamlit
- [ ] Script CLI `ingest.py` pour alimenter la base sans relancer l'app

**Livrable :** L'assistant répond en citant les documents BIM internes. ✅

---

### Phase 3 — Qualité & robustesse (Semaine 5)
> *"Make it right."*

- [ ] Découpage en modules propres (`src/retriever.py`, `src/llm.py`, `src/ui.py`)
- [ ] Logging structuré : questions posées, documents retrouvés, latence HF
- [ ] Tests unitaires sur le pipeline RAG (`pytest`)
- [ ] Fichier de config centralisé (`config.yaml` : chunk_size, top_k, modèle HF)
- [ ] `requirements.txt` verrouillé (`pip freeze > requirements.txt`)

**Livrable :** Code maintenable, tests qui passent, config externalisée. ✅

---

### Phase 4 — Fonctionnalités métier avancées (Semaines 6–8)
> *"Make it useful."*

- [ ] **Sélecteur de projet** : chaque projet a sa propre collection ChromaDB
- [ ] **Upload de document** directement depuis l'UI Streamlit (ingestion à la volée)
- [ ] **Mode équipe** : filtrer les documents par équipe (Modélisation / MEP / Travaux)
- [ ] Parsing IFC basique avec `ifcopenshell` (extraction de métadonnées éléments)
- [ ] Résumé automatique de compte-rendu de réunion (upload PDF → résumé structuré)

**Livrable :** L'assistant est utilisable par les équipes sur des projets réels. ✅

---

### Phase 5 — Déploiement interne (Semaine 9+)
> *"Make it available."*

- [ ] Dockerisation de l'application (`Dockerfile` + `docker-compose.yml`)
- [ ] Déploiement sur serveur interne ou Streamlit Community Cloud
- [ ] Authentification simple (mot de passe via `st.secrets`)
- [ ] Documentation utilisateur (guide PDF d'onboarding pour les équipes)

**Livrable :** URL interne accessible par toutes les équipes BIM. ✅

---

## Stack technique

| Couche | Outil |
|---|---|
| Interface | Streamlit |
| LLM | `mistralai/Mistral-7B-Instruct-v0.3` via `HuggingFaceEndpoint` |
| Orchestration RAG | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, gratuit) |
| Base vectorielle | ChromaDB (local) |
| Parseur IFC | ifcopenshell |
| Tests | pytest |
| Containerisation | Docker |

---

##  Structure du projet

```
bim-assistant-btp/
│
├── data/
│   ├── raw/               # Documents sources (PDF, DOCX) — non versionné
│   └── chroma_db/         # Base vectorielle persistante — non versionné
│
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Pipeline ingestion documents → ChromaDB
│   ├── retriever.py       # Recherche vectorielle (ChromaDB + MiniLM)
│   ├── llm.py             # HuggingFaceEndpoint + construction du prompt
│   └── ui.py              # Composants Streamlit réutilisables
│
├── tests/
│   ├── test_retriever.py
│   └── test_llm.py
│
├── app.py                 # Point d'entrée Streamlit
├── config.yaml            # Paramètres (chunk_size, top_k, modèle HF...)
├── .env.example           # Template variables d'environnement
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

---

##  Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/<org>/bim-assistant-btp.git
cd bim-assistant-btp

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Variables d'environnement
cp .env.example .env
# → Renseigner HUGGINGFACEHUB_API_TOKEN dans .env

# 5. Ingérer les documents BIM
python src/ingest.py --source data/raw/

# 6. Lancer l'application
streamlit run app.py
```

---

## Configuration (`config.yaml`)

```yaml
llm:
  repo_id: mistralai/Mistral-7B-Instruct-v0.3
  max_new_tokens: 1024
  temperature: 0.2
  task: text-generation

retriever:
  chunk_size: 800
  chunk_overlap: 100
  top_k: 5
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

chroma:
  persist_directory: data/chroma_db
  collection_name: bim_docs
```

---

##  Variables d'environnement (`.env.example`)

```env
# Hugging Face — générer sur https://huggingface.co/settings/tokens
HUGGINGFACEHUB_API_TOKEN=hf_...
```

---

##  Dépendances principales (`requirements.txt`)

```
streamlit
langchain
langchain-huggingface
langchain-community
chromadb
sentence-transformers
pypdf
python-docx
ifcopenshell
pyyaml
python-dotenv
pytest
```

---

##  Conventions Git

```
feat: ajout du pipeline RAG
fix: correction du chunking PDF
docs: mise à jour README
refactor: découpage src/llm.py
test: ajout tests retriever
chore: mise à jour requirements.txt
```

---

##  Limites Hugging Face Inference API (tier gratuit)

| Limite | Valeur |
|---|---|
| Requêtes / heure | ~300 |
| Taille contexte | ~4 096 tokens |
| Modèles disponibles | Tous les modèles publics |
| Disponibilité | Partagée (cold start possible) |

> Pour un usage prod intensif : passer sur **Hugging Face Inference Endpoints** (instance dédiée) ou **Ollama** en local sur un serveur interne.

---

##  Licence

Usage interne — propriété de l'entreprise. Ne pas distribuer.