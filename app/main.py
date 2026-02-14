"""
RAG Portfolio — API FastAPI
Endpoints : question/réponse, indexation, évaluation, comparaison de modèles
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict
import logging
from datetime import datetime

from rag.pipeline import RAGPipeline, EMBEDDING_MODELS, DEFAULT_MODEL
from rag.data import get_sample_documents, get_sample_qa_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline global
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initialisation du pipeline RAG...")
    pipeline = RAGPipeline(embedding_model=DEFAULT_MODEL)

    # Indexe les documents de démonstration si la base est vide
    if pipeline.vector_store.collection.count() == 0:
        docs = get_sample_documents()
        n = pipeline.index_documents(docs)
        logger.info(f"{n} chunks indexés au démarrage")

    logger.info(f"Pipeline RAG prêt — modèle : {DEFAULT_MODEL} ✓")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="RAG Portfolio API",
    description="Système RAG spécialisé IA/ML avec comparaison de modèles d'embeddings",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Schémas ─────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    n_chunks: int = 5
    embedding_model: Optional[str] = DEFAULT_MODEL

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    similarity_scores: List[float]
    embedding_model: str
    retrieval_time_ms: float
    total_time_ms: float
    n_chunks_retrieved: int

class IndexRequest(BaseModel):
    documents: List[Dict]

class EvaluateRequest(BaseModel):
    models: Optional[List[str]] = None
    n_questions: int = 5

class DocumentInput(BaseModel):
    title: str
    content: str
    source: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "title": "RAG Portfolio API 🤖",
        "description": "Système RAG spécialisé papers IA/ML",
        "embedding_models": list(EMBEDDING_MODELS.keys()),
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "n_documents_indexed": pipeline.vector_store.collection.count() if pipeline else 0,
        "embedding_model": DEFAULT_MODEL,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    Pose une question au système RAG.
    Le système récupère les passages pertinents et génère une réponse sourcée.
    """
    if pipeline is None:
        raise HTTPException(503, "Pipeline non initialisé")

    if not request.question.strip():
        raise HTTPException(400, "La question ne peut pas être vide")

    response = pipeline.query(request.question, n_chunks=request.n_chunks)

    return QuestionResponse(
        question=response.question,
        answer=response.answer,
        sources=list(set([c.source for c in response.retrieved_chunks])),
        similarity_scores=[c.similarity_score for c in response.retrieved_chunks],
        embedding_model=response.embedding_model,
        retrieval_time_ms=response.retrieval_time_ms,
        total_time_ms=response.total_time_ms,
        n_chunks_retrieved=response.n_chunks_retrieved
    )

@app.post("/index")
def index_documents(request: IndexRequest):
    """Indexe de nouveaux documents dans la base vectorielle."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline non initialisé")

    n_chunks = pipeline.index_documents(request.documents)
    return {
        "status": "success",
        "chunks_indexed": n_chunks,
        "total_in_db": pipeline.vector_store.collection.count()
    }

@app.get("/evaluate")
def evaluate_models(models: Optional[str] = None, n_questions: int = 3):
    """
    Compare les modèles d'embeddings sur un dataset de questions de référence.
    C'est la fonctionnalité différenciante du projet.
    """
    from evaluation.metrics import ModelComparator

    model_list = models.split(",") if models else list(EMBEDDING_MODELS.keys())[:2]
    qa_pairs = get_sample_qa_pairs()[:n_questions]
    documents = get_sample_documents()

    comparator = ModelComparator()
    report = comparator.compare_models(documents, qa_pairs, models=model_list)

    return {
        "models_evaluated": report.models_evaluated,
        "avg_scores": report.avg_scores,
        "best_model": report.best_model,
        "n_questions": report.n_questions,
        "evaluation_date": report.evaluation_date,
        "interpretation": {
            "best_model": f"{report.best_model} est le meilleur modèle d'embeddings pour ce corpus",
            "score_range": "0 = mauvais, 1 = parfait",
            "metrics": "Moyenne de faithfulness, answer_relevancy, context_precision, context_recall"
        }
    }

@app.get("/models")
def list_models():
    """Liste les modèles d'embeddings disponibles avec leurs caractéristiques."""
    return {
        "available_models": {
            "minilm": {
                "model_id": EMBEDDING_MODELS["minilm"],
                "dimensions": 384,
                "speed": "fast",
                "description": "Rapide et léger, bon équilibre vitesse/qualité"
            },
            "mpnet": {
                "model_id": EMBEDDING_MODELS["mpnet"],
                "dimensions": 768,
                "speed": "medium",
                "description": "Meilleure qualité, recommandé pour la production"
            },
            "scibert": {
                "model_id": EMBEDDING_MODELS["scibert"],
                "dimensions": 768,
                "speed": "medium",
                "description": "Spécialisé textes scientifiques, optimal pour papers IA/ML"
            }
        },
        "current_model": DEFAULT_MODEL
    }

@app.get("/stats")
def get_stats():
    """Statistiques sur la base de connaissances."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline non initialisé")

    n_docs = pipeline.vector_store.collection.count()
    return {
        "total_chunks_indexed": n_docs,
        "embedding_model": DEFAULT_MODEL,
        "embedding_dimensions": pipeline.embedding_engine.embedding_dim,
        "vector_db": "ChromaDB",
        "llm_provider": pipeline.llm_client.provider,
    }
