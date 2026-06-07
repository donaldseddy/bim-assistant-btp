# 🏗️ BIM Assistant BTP
 
> Chatbot interne RAG pour les équipes BIM — modélisation, coordination, travaux, MEP.  
> Propulsé par **Mistral 7B Instruct** via Hugging Face — 100% open source, zéro coût LLM.
 

##  Contexte & Problématique
 
### Le quotidien des équipes BIM en entreprise BTP
 
Dans une entreprise de construction, les équipes BIM produisent et consomment quotidiennement un volume massif de documentation : BEP (BIM Execution Plan), DTU, plans d'exécution, fiches techniques fabricants, comptes-rendus de coordination, clash reports, conventions de nommage, LOD par corps d'état, etc.
 
Ce volume génère trois problèmes concrets et récurrents :
 
---
 
### ❌ Problème 1 — La connaissance est silotée
 
Chaque équipe (Modélisation, MEP, Travaux, BIM Management) accumule sa propre documentation dans des répertoires partagés mal organisés. Un ingénieur MEP qui cherche la convention de nommage des gaines CVC sur le projet X doit :
 
1. Retrouver où est stocké le BEP du projet
2. Ouvrir un PDF de 80 pages
3. Chercher manuellement la section pertinente
**Résultat :** Perte de temps, erreurs de modélisation, re-travail coûteux.
 
---
 
### ❌ Problème 2 — Les standards internes ne sont pas accessibles en temps réel
 
Les chartes BIM, gabarits Revit, guides MEP, bibliothèques de familles — ces ressources existent mais sont dispersées. Les nouveaux arrivants (alternants, sous-traitants, nouvelles recrues) passent des jours à trouver les bons documents, ou reproduisent des erreurs faute d'accès rapide aux standards.
 
**Résultat :** Qualité de modélisation hétérogène selon les profils, non-conformités détectées tard.
 
---
 
### ❌ Problème 3 — Les normes BIM sont complexes et peu mémorisées
 
ISO 19650, EN 17412, PPBIM, LOD vs LOI, CDE, PIR vs AIR... Les équipes ont besoin de réponses précises sur des normes en constante évolution. Recourir à un moteur de recherche généraliste (Google, ChatGPT) ne donne pas de réponses adaptées au contexte de l'entreprise.
 
**Résultat :** Interprétations divergentes entre équipes, risque de non-conformité contractuelle.
 
---
 
### 🎯 Question centrale
 
> **Comment permettre à n'importe quel membre d'une équipe BIM d'interroger en langage naturel la base documentaire interne de l'entreprise — et obtenir une réponse précise, sourcée, en quelques secondes ?**
 
---
 
##  Solution proposée
 
**BIM Assistant BTP** est un chatbot interne basé sur l'architecture **RAG (Retrieval-Augmented Generation)**.
 
Le principe : au lieu de demander au modèle de langue de "mémoriser" les documents de l'entreprise (impossible), on lui injecte dynamiquement les passages les plus pertinents au moment de chaque question — comme si on glissait les bonnes pages d'un manuel sous les yeux d'un expert avant qu'il réponde.
 
```
Question utilisateur
       ↓
  [Recherche vectorielle]
  → Retrouve les 5 passages les plus proches dans ChromaDB
       ↓
  [Construction du prompt]
  → Contexte documentaire + question + system prompt BIM
       ↓
  [Mistral 7B Instruct — HuggingFace]
  → Génère une réponse ancrée dans les documents
       ↓
  Réponse + sources citées affichées dans Streamlit
```
 
**Avantages clés :**
- Les réponses sont toujours ancrées dans les documents réels de l'entreprise
- Les sources sont citées → traçabilité et confiance
- Aucun coût LLM (Mistral 7B open source via HuggingFace)
- Les données confidentielles ne quittent pas l'infra (phase Ollama)

## ⚠️ Limites & évolutions
 
### Limites actuelles
 
| Limite | Impact | Mitigation prévue |
|---|---|---|
| HF Inference API partagée | Latence variable (cold start) | Phase 5 : Ollama sur serveur interne |
| Mistral 7B contexte ~4K tokens | Réponses tronquées sur longs docs | Chunking optimisé (Expérience 2) |
| ChromaDB local | Non scalable multi-utilisateurs | Migration Pinecone ou Qdrant |
| Pas d'auth utilisateur | Accès non contrôlé | Phase 5 : `st.secrets` ou SSO |
 
### Évolutions planifiées
 
- [ ] **Ollama** en local pour les données ultra-confidentielles (zéro sortie réseau)
- [ ] **Connexion BIM 360 / ACC** via Autodesk Platform Services API
- [ ] **Parsing IFC** avec `ifcopenshell` (extraction métrés, métadonnées éléments)
- [ ] **Add-in Revit** pour interroger l'assistant sans quitter l'environnement de modélisation
- [ ] **Multi-collection** : une collection ChromaDB par projet
- [ ] **Streaming** : afficher la réponse au fur et à mesure de sa génération
