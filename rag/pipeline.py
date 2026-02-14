"""
RAG Engine — Retrieval-Augmented Generation
Domaine : Papers scientifiques en IA/ML (arXiv)
Auteur  : [Ton Prénom Nom]

Fonctionnement :
1. Les documents sont découpés en chunks et encodés en vecteurs (embeddings)
2. La question de l'utilisateur est encodée de la même façon
3. On cherche les chunks les plus similaires à la question (recherche vectorielle)
4. On envoie ces chunks + la question au LLM pour générer une réponse sourcée
"""

import os
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ─── Modèles d'embeddings disponibles ────────────────────────────────────────
# C'est ici le "twist" du projet : on compare plusieurs modèles d'embeddings
EMBEDDING_MODELS = {
    "minilm":    "all-MiniLM-L6-v2",        # Rapide, léger (384 dims)
    "mpnet":     "all-mpnet-base-v2",         # Meilleur, plus lent (768 dims)
    "scibert":   "allenai/scibert_scivocab_uncased",  # Spécialisé scientifique
}

DEFAULT_MODEL = "minilm"


@dataclass
class RetrievedChunk:
    """Un chunk de document récupéré avec son score de similarité."""
    content: str
    source: str
    chunk_id: str
    similarity_score: float
    metadata: Dict


@dataclass
class RAGResponse:
    """Réponse complète du système RAG avec métriques."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    embedding_model: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    n_chunks_retrieved: int


class DocumentProcessor:
    """Découpe les documents en chunks avec overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Découpe un texte en chunks avec métadonnées."""
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:  # Ignore les chunks trop petits
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_index": len(chunks),
                "word_count": len(chunk_words),
            })

        logger.info(f"Document '{source}' → {len(chunks)} chunks")
        return chunks

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Traite une liste de documents {title, content, source}."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["content"], doc.get("source", doc["title"]))
            for chunk in chunks:
                chunk["doc_title"] = doc["title"]
            all_chunks.extend(chunks)
        return all_chunks


class VectorStore:
    """
    Base de données vectorielle avec ChromaDB.
    Stocke les embeddings et permet la recherche par similarité cosinus.
    """

    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Distance cosinus pour les embeddings texte
        )
        logger.info(f"VectorStore initialisé — {self.collection.count()} documents existants")

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """Ajoute des chunks avec leurs embeddings dans ChromaDB."""
        if len(chunks) == 0:
            return

        ids = [f"chunk_{i}_{int(time.time())}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        metadatas = [{
            "source": c["source"],
            "doc_title": c.get("doc_title", ""),
            "chunk_index": c["chunk_index"],
        } for c in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"{len(chunks)} chunks ajoutés. Total : {self.collection.count()}")

    def search(self, query_embedding: np.ndarray, n_results: int = 5) -> List[RetrievedChunk]:
        """Recherche les chunks les plus similaires à la requête."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - dist  # ChromaDB retourne une distance, on convertit en similarité
            chunks.append(RetrievedChunk(
                content=doc,
                source=meta.get("source", "unknown"),
                chunk_id=results["ids"][0][i],
                similarity_score=round(similarity, 4),
                metadata=meta
            ))

        return chunks

    def clear(self):
        """Vide la collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


class EmbeddingEngine:
    """Gère les modèles d'embeddings — permet la comparaison entre modèles."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        model_id = EMBEDDING_MODELS.get(model_name, EMBEDDING_MODELS[DEFAULT_MODEL])
        logger.info(f"Chargement du modèle d'embeddings : {model_id}")
        self.model = SentenceTransformer(model_id)
        self.model_name = model_name
        self.model_id = model_id
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode une liste de textes en vecteurs."""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode une seule requête."""
        return self.model.encode([query])[0]


class LLMClient:
    """
    Client LLM flexible — supporte Mistral AI et OpenAI.
    Fallback sur une réponse template si pas de clé API.
    """

    def __init__(self):
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self.mistral_key:
            return "mistral"
        elif self.openai_key:
            return "openai"
        return "fallback"

    def generate(self, question: str, context_chunks: List[RetrievedChunk]) -> str:
        """Génère une réponse à partir de la question et du contexte récupéré."""
        context = "\n\n".join([
            f"[Source: {c.source}]\n{c.content}"
            for c in context_chunks
        ])

        prompt = f"""Tu es un assistant expert en intelligence artificielle et machine learning.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Cite tes sources entre parenthèses.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""

        if self.provider == "mistral":
            return self._call_mistral(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._fallback_response(question, context_chunks)

    def _call_mistral(self, prompt: str) -> str:
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            client = MistralClient(api_key=self.mistral_key)
            response = client.chat(
                model="mistral-small-latest",
                messages=[ChatMessage(role="user", content=prompt)]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erreur Mistral: {e}")
            return f"Erreur API Mistral: {str(e)}"

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erreur OpenAI: {e}")
            return f"Erreur API OpenAI: {str(e)}"

    def _fallback_response(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Réponse sans LLM — utile pour les tests sans clé API."""
        if not chunks:
            return "Aucun document pertinent trouvé pour répondre à cette question."

        best = chunks[0]
        return (
            f"[Mode démonstration — sans LLM]\n\n"
            f"Question : {question}\n\n"
            f"Extrait le plus pertinent (similarité : {best.similarity_score:.2%}) :\n"
            f"{best.content[:500]}...\n\n"
            f"Source : {best.source}"
        )


class RAGPipeline:
    """
    Pipeline RAG complet.
    Orchestre : DocumentProcessor → EmbeddingEngine → VectorStore → LLMClient
    """

    def __init__(self, embedding_model: str = DEFAULT_MODEL, persist_dir: str = "./chroma_db"):
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.doc_processor = DocumentProcessor()
        self.llm_client = LLMClient()

    def index_documents(self, documents: List[Dict]) -> int:
        """Indexe des documents dans la base vectorielle."""
        chunks = self.doc_processor.process_documents(documents)
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = self.embedding_engine.encode(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

    def query(self, question: str, n_chunks: int = 5) -> RAGResponse:
        """Répond à une question en utilisant le pipeline RAG complet."""
        t_start = time.time()

        # 1. Encode la question
        query_embedding = self.embedding_engine.encode_query(question)
        t_retrieved = time.time()

        # 2. Récupère les chunks les plus pertinents
        chunks = self.vector_store.search(query_embedding, n_results=n_chunks)
        retrieval_time = (time.time() - t_retrieved) * 1000

        # 3. Génère la réponse avec le LLM
        t_gen = time.time()
        answer = self.llm_client.generate(question, chunks)
        generation_time = (time.time() - t_gen) * 1000

        total_time = (time.time() - t_start) * 1000

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            embedding_model=self.embedding_engine.model_name,
            retrieval_time_ms=round(retrieval_time, 2),
            generation_time_ms=round(generation_time, 2),
            total_time_ms=round(total_time, 2),
            n_chunks_retrieved=len(chunks)
        )
