#  RAG Portfolio — Système de Question-Réponse sur Papers IA/ML

> Système RAG (Retrieval-Augmented Generation) spécialisé sur des papers scientifiques IA/ML.
> **Différenciateur clé** : comparaison rigoureuse de 3 modèles d'embeddings avec métriques RAGAS.

![CI](https://github.com/Rayanmlk/rag-portfolio/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)

## 🎯 Ce que fait ce projet

Pose une question sur les Transformers, BERT, GPT-3, ou les systèmes RAG → l'API récupère les passages pertinents dans une base vectorielle de papers et génère une réponse sourcée et vérifiable.

**Ce qui le distingue d'un simple chatbot** : le système évalue et compare 3 modèles d'embeddings différents avec 4 métriques objectives (Faithfulness, Answer Relevancy, Context Precision, Context Recall).

## 🏗️ Architecture

```
Question utilisateur
        ↓
[Embedding Engine] → encode la question en vecteur
        ↓
[ChromaDB VectorStore] → recherche les K passages les plus similaires
        ↓
[LLM Client] → génère une réponse à partir des passages (Mistral/OpenAI)
        ↓
Réponse sourcée + métriques temps réel
```

```
rag-portfolio/
├── app/
│   └── main.py          # API FastAPI (ask, index, evaluate, models, stats)
├── rag/
│   ├── pipeline.py      # RAGPipeline, EmbeddingEngine, VectorStore, LLMClient
│   └── data.py          # Documents de démo (papers IA/ML) + QA pairs
├── evaluation/
│   └── metrics.py       # RAGEvaluator + ModelComparator
├── tests/
│   └── test_all.py      # Tests unitaires et d'intégration
└── .github/workflows/
    └── ci.yml           # Tests → Évaluation RAG → Deploy
```

## ⚡ Démarrage rapide

```bash
git clone https://github.com/TON_USERNAME/rag-portfolio
cd rag-portfolio
pip install -r requirements.txt

# Optionnel : clé API pour LLM (sans clé = mode démo)
export MISTRAL_API_KEY=your_key  # ou OPENAI_API_KEY

uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

## 📡 Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Info + modèles disponibles |
| GET | `/health` | Santé + nb documents indexés |
| POST | `/ask` | **Question → Réponse sourcée** |
| POST | `/index` | Indexer de nouveaux documents |
| GET | `/evaluate` | **Comparer les modèles d'embeddings** |
| GET | `/models` | Détails des modèles disponibles |
| GET | `/stats` | Stats de la base vectorielle |

### Exemple de question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer architecture?", "n_chunks": 5}'
```

```json
{
  "answer": "The Transformer uses self-attention mechanisms without recurrence...",
  "sources": ["Vaswani et al., 2017 — arXiv:1706.03762"],
  "similarity_scores": [0.87, 0.82, 0.79],
  "retrieval_time_ms": 12.4,
  "embedding_model": "minilm"
}
```

## 🔬 Comparaison des modèles d'embeddings

```bash
GET /evaluate?models=minilm,mpnet&n_questions=5
```

```json
{
  "avg_scores": {
    "minilm": 0.6234,
    "mpnet":  0.7012
  },
  "best_model": "mpnet",
  "interpretation": {
    "best_model": "mpnet est le meilleur modèle pour ce corpus"
  }
}
```

**4 métriques évaluées :**
- **Faithfulness** (30%) — la réponse s'appuie-t-elle sur les sources ?
- **Answer Relevancy** (30%) — la réponse répond-elle à la question ?
- **Context Precision** (20%) — les passages récupérés sont-ils pertinents ?
- **Context Recall** (20%) — les passages contiennent-ils l'information nécessaire ?

## 🧠 Modèles d'embeddings comparés

| Modèle | Dims | Vitesse | Usage optimal |
|--------|------|---------|---------------|
| `minilm` | 384 | ⚡ Rapide | Prototypage, faible latence |
| `mpnet` | 768 | 🔄 Moyen | Production généraliste |
| `scibert` | 768 | 🔄 Moyen | Textes scientifiques |

## 🧪 Tests

```bash
pytest tests/ -v --cov=app --cov=rag --cov=evaluation
```

## 🛠️ Stack

`Python` · `FastAPI` · `ChromaDB` · `Sentence-Transformers` · `Mistral AI` · `Docker` · `GitHub Actions`

## 👤 Auteur

**[Ton Prénom Nom]** — M1 Data & IA
- GitHub: [@Rayanmlk](https://github.com/TON_USERNAME)
- Demo: [API live](https://rag-portfolio.railway.app/docs)
