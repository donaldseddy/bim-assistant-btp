"""
Phase 0 — Script de validation de la connexion HuggingFace.
─────────────────────────────────────────────────────────────
Lancer AVANT de coder quoi que ce soit d'autre.
Commande : python test_llm.py

Si ce script échoue → vérifier HUGGINGFACEHUB_API_TOKEN dans .env.
Si ce script passe  → passer à la Phase 1 (app.py).
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token or not token.startswith("hf_"):
    print("❌ HUGGINGFACEHUB_API_TOKEN manquant ou invalide dans .env")
    sys.exit(1)

print("✅ Token HuggingFace détecté")
print("⏳ Connexion à Mistral-7B-Instruct-v0.3...")

try:
    from langchain_huggingface import HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=256,
        temperature=0.2,
        huggingfacehub_api_token=token,
    )

    question = "En une phrase, qu'est-ce qu'un BEP en BIM ?"
    print(f"\n🔹 Question test : {question}")

    reponse = llm.invoke(question)
    print(f"✅ Réponse reçue :\n{reponse}")
    print("\n✅ Phase 0 validée — tu peux passer à la Phase 1.")

except Exception as e:
    print(f"❌ Erreur lors de l'appel HuggingFace : {e}")
    sys.exit(1)
