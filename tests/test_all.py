"""
Tests — RAG Portfolio
pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import DocumentProcessor, RAGPipeline, EmbeddingEngine
from rag.data import get_sample_documents, get_sample_qa_pairs
from evaluation.metrics import RAGEvaluator, RAGMetrics
from fastapi.testclient import TestClient


# ─── Tests DocumentProcessor ─────────────────────────────────────────────────

class TestDocumentProcessor:

    def test_chunk_basic(self):
        proc = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = proc.chunk_text(text, source="test")
        assert len(chunks) > 1
        assert all("content" in c for c in chunks)
        assert all("source" in c for c in chunks)

    def test_chunk_source_preserved(self):
        proc = DocumentProcessor()
        text = " ".join(["hello"] * 100)
        chunks = proc.chunk_text(text, source="my_paper.pdf")
        assert all(c["source"] == "my_paper.pdf" for c in chunks)

    def test_chunk_overlap(self):
        proc = DocumentProcessor(chunk_size=20, chunk_overlap=5)
        text = " ".join([f"w{i}" for i in range(100)])
        chunks = proc.chunk_text(text, source="test")
        assert len(chunks) >= 2

    def test_process_multiple_documents(self):
        proc = DocumentProcessor()
        docs = get_sample_documents()[:2]
        chunks = proc.process_documents(docs)
        assert len(chunks) > 0
        assert all("doc_title" in c for c in chunks)


# ─── Tests EmbeddingEngine ────────────────────────────────────────────────────

class TestEmbeddingEngine:

    def test_encode_single(self):
        engine = EmbeddingEngine("minilm")
        embedding = engine.encode_query("What is machine learning?")
        assert embedding.shape[0] == 384  # minilm = 384 dims
        assert abs(embedding).max() > 0   # pas un vecteur nul

    def test_encode_batch(self):
        engine = EmbeddingEngine("minilm")
        texts = ["Hello world", "Machine learning is great", "RAG systems work well"]
        embeddings = engine.encode(texts)
        assert embeddings.shape == (3, 384)

    def test_similarity_meaningful(self):
        """Deux phrases similaires doivent avoir une similarité plus élevée."""
        import numpy as np
        engine = EmbeddingEngine("minilm")
        e1 = engine.encode_query("Transformer attention mechanism")
        e2 = engine.encode_query("Self-attention in transformers")
        e3 = engine.encode_query("Pizza recipe with tomatoes")

        # Similarité cosinus
        sim_related = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        sim_unrelated = np.dot(e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3))

        assert sim_related > sim_unrelated, "Phrases similaires doivent être plus proches"


# ─── Tests RAGEvaluator ───────────────────────────────────────────────────────

class TestRAGEvaluator:

    def test_faithfulness_high_overlap(self):
        evaluator = RAGEvaluator()

        class MockChunk:
            content = "machine learning transformers attention neural networks"
            source = "test"

        score = evaluator.compute_faithfulness(
            "The transformer uses attention mechanisms in neural networks",
            [MockChunk()]
        )
        assert score > 0.3

    def test_faithfulness_no_overlap(self):
        evaluator = RAGEvaluator()

        class MockChunk:
            content = "pizza recipe tomato sauce mozzarella basil"
            source = "test"

        score = evaluator.compute_faithfulness(
            "Transformers use attention mechanisms",
            [MockChunk()]
        )
        assert score < 0.5

    def test_answer_relevancy_empty(self):
        evaluator = RAGEvaluator()
        score = evaluator.compute_answer_relevancy("What is BERT?", "")
        assert score < 0.3

    def test_answer_relevancy_good(self):
        evaluator = RAGEvaluator()
        score = evaluator.compute_answer_relevancy(
            "What is BERT pre-training?",
            "BERT pre-training uses Masked Language Modeling and Next Sentence Prediction "
            "on large corpora to learn bidirectional representations."
        )
        assert score > 0.3

    def test_metrics_all_present(self):
        evaluator = RAGEvaluator()

        class MockChunk:
            content = "BERT uses masked language modeling for pre-training"
            source = "test"
            similarity_score = 0.85

        metrics = evaluator.compute_metrics(
            question="How does BERT train?",
            answer="BERT trains using masked language modeling",
            expected_answer="BERT uses masked language modeling",
            chunks=[MockChunk()]
        )
        assert 0 <= metrics.faithfulness <= 1
        assert 0 <= metrics.answer_relevancy <= 1
        assert 0 <= metrics.context_precision <= 1
        assert 0 <= metrics.context_recall <= 1
        assert 0 <= metrics.overall_score <= 1


# ─── Tests données de démo ────────────────────────────────────────────────────

class TestSampleData:

    def test_documents_structure(self):
        docs = get_sample_documents()
        assert len(docs) >= 3
        for doc in docs:
            assert "title" in doc
            assert "content" in doc
            assert len(doc["content"]) > 100

    def test_qa_pairs_structure(self):
        qa_pairs = get_sample_qa_pairs()
        assert len(qa_pairs) >= 3
        for qa in qa_pairs:
            assert "question" in qa
            assert "expected_answer" in qa
            assert len(qa["question"]) > 10


# ─── Tests API ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestAPI:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "embedding_models" in data

    def test_models_endpoint(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "available_models" in data
        assert "minilm" in data["available_models"]

    def test_stats_endpoint(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_chunks_indexed" in data
        assert data["total_chunks_indexed"] >= 0

    def test_ask_question(self, client):
        r = client.post("/ask", json={
            "question": "What is the Transformer architecture?",
            "n_chunks": 3
        })
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieval_time_ms" in data
        assert len(data["answer"]) > 10

    def test_ask_empty_question(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 400

    def test_index_documents(self, client):
        r = client.post("/index", json={
            "documents": [{
                "title": "Test Document",
                "content": "This is a test document about machine learning and deep learning " * 10,
                "source": "test_source"
            }]
        })
        assert r.status_code == 200
        assert r.json()["chunks_indexed"] > 0
