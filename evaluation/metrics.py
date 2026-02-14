"""
Évaluation RAG avec métriques RAGAS-inspired
C'est le "twist" du projet : on ne fait pas que construire le RAG,
on mesure RIGOUREUSEMENT sa qualité avec des métriques standard.

Métriques implémentées :
- Faithfulness    : la réponse est-elle fidèle aux sources ?
- Answer Relevancy: la réponse répond-elle à la question ?
- Context Recall  : les bons documents ont-ils été récupérés ?
- Context Precision: les documents récupérés sont-ils pertinents ?
"""

import numpy as np
import time
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

from rag.pipeline import RAGPipeline, RAGResponse, EMBEDDING_MODELS

logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """Métriques d'évaluation pour une réponse RAG."""
    faithfulness: float          # 0-1 : réponse cohérente avec les sources
    answer_relevancy: float      # 0-1 : réponse pertinente pour la question
    context_precision: float     # 0-1 : chunks récupérés vraiment utiles
    context_recall: float        # 0-1 : tous les chunks utiles récupérés
    overall_score: float         # moyenne pondérée

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Résultat complet pour une paire question/réponse."""
    question: str
    expected_answer: str
    generated_answer: str
    embedding_model: str
    metrics: RAGMetrics
    retrieval_time_ms: float
    generation_time_ms: float
    n_chunks: int


@dataclass
class ModelComparisonReport:
    """Rapport de comparaison entre plusieurs modèles d'embeddings."""
    models_evaluated: List[str]
    results_per_model: Dict[str, List[EvaluationResult]]
    avg_scores: Dict[str, float]
    best_model: str
    evaluation_date: str
    n_questions: int


class RAGEvaluator:
    """
    Évalue la qualité d'un pipeline RAG avec des métriques objectives.
    Implémentation simplifiée mais rigoureuse des métriques RAGAS.
    """

    def __init__(self, embedding_engine=None):
        """
        embedding_engine : optionnel, pour calculer des similarités sémantiques.
        Sans LLM de jugement, on utilise des heuristiques robustes.
        """
        self.embedding_engine = embedding_engine

    def compute_faithfulness(self, answer: str, context_chunks) -> float:
        """
        Faithfulness : mesure si la réponse s'appuie sur les sources.
        Heuristique : overlap entre mots-clés de la réponse et du contexte.
        """
        if not context_chunks or not answer:
            return 0.0

        context_text = " ".join([c.content.lower() for c in context_chunks])
        answer_words = set(answer.lower().split())

        # Filtre les stop words
        stop_words = {
            "le", "la", "les", "un", "une", "des", "de", "du", "et", "en",
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
            "for", "that", "this", "with", "it", "its"
        }
        meaningful_words = answer_words - stop_words

        if not meaningful_words:
            return 0.5

        found = sum(1 for w in meaningful_words if w in context_text)
        return round(min(found / len(meaningful_words), 1.0), 4)

    def compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy : la réponse répond-elle à la question ?
        Heuristique : overlap de mots-clés question/réponse + longueur.
        """
        if not answer or len(answer) < 20:
            return 0.1

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        stop_words = {"le", "la", "les", "un", "de", "est", "the", "a", "is", "what", "how", "why"}
        q_keywords = question_words - stop_words
        a_keywords = answer_words - stop_words

        if not q_keywords:
            return 0.5

        overlap = len(q_keywords & a_keywords) / len(q_keywords)

        # Bonus si la réponse est substantielle
        length_bonus = min(len(answer.split()) / 100, 0.2)

        # Malus si le modèle dit qu'il ne sait pas
        uncertainty_words = ["ne sais pas", "don't know", "cannot", "no information", "aucune"]
        uncertainty_penalty = 0.3 if any(w in answer.lower() for w in uncertainty_words) else 0

        score = min(overlap + length_bonus - uncertainty_penalty, 1.0)
        return round(max(score, 0.0), 4)

    def compute_context_precision(self, question: str, chunks) -> float:
        """
        Context Precision : parmi les chunks récupérés, combien sont vraiment utiles ?
        """
        if not chunks:
            return 0.0

        question_words = set(question.lower().split())
        stop_words = {"le", "la", "les", "de", "est", "the", "a", "is", "what", "how"}
        q_keywords = question_words - stop_words

        if not q_keywords:
            return 0.5

        relevant_chunks = 0
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(q_keywords & chunk_words) / len(q_keywords)
            if overlap > 0.15:  # Au moins 15% des mots-clés présents
                relevant_chunks += 1

        return round(relevant_chunks / len(chunks), 4)

    def compute_context_recall(self, expected_answer: str, chunks) -> float:
        """
        Context Recall : les sources contiennent-elles l'info pour répondre ?
        Proxy : les mots-clés de la réponse attendue sont-ils dans les sources ?
        """
        if not chunks or not expected_answer:
            return 0.0

        expected_words = set(expected_answer.lower().split())
        stop_words = {"le", "la", "les", "de", "est", "the", "a", "is", "and", "or"}
        expected_keywords = expected_words - stop_words

        if not expected_keywords:
            return 0.5

        context_text = " ".join([c.content.lower() for c in chunks])
        found = sum(1 for w in expected_keywords if w in context_text)
        return round(min(found / len(expected_keywords), 1.0), 4)

    def compute_metrics(
        self,
        question: str,
        answer: str,
        expected_answer: str,
        chunks
    ) -> RAGMetrics:
        """Calcule toutes les métriques pour une réponse."""
        faithfulness     = self.compute_faithfulness(answer, chunks)
        answer_relevancy = self.compute_answer_relevancy(question, answer)
        context_precision = self.compute_context_precision(question, chunks)
        context_recall   = self.compute_context_recall(expected_answer, chunks)

        # Moyenne pondérée (faithfulness et relevancy comptent plus)
        overall = (
            faithfulness     * 0.30 +
            answer_relevancy * 0.30 +
            context_precision * 0.20 +
            context_recall   * 0.20
        )

        return RAGMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=round(overall, 4)
        )


class ModelComparator:
    """
    Compare plusieurs modèles d'embeddings sur un dataset de questions.
    C'est la pièce maîtresse du projet — ce qui le distingue d'un simple chatbot.
    """

    def __init__(self, persist_base_dir: str = "./chroma_db"):
        self.persist_base_dir = persist_base_dir
        self.evaluator = RAGEvaluator()

    def evaluate_model(
        self,
        model_name: str,
        documents: List[Dict],
        qa_pairs: List[Dict],
        n_chunks: int = 5
    ) -> Tuple[List[EvaluationResult], float]:
        """Évalue un modèle d'embeddings sur un set de questions/réponses."""
        from datetime import datetime

        logger.info(f"\n{'='*50}")
        logger.info(f"Évaluation du modèle : {model_name}")
        logger.info(f"{'='*50}")

        persist_dir = f"{self.persist_base_dir}_{model_name}"
        pipeline = RAGPipeline(embedding_model=model_name, persist_dir=persist_dir)

        # Indexation des documents
        n_indexed = pipeline.index_documents(documents)
        logger.info(f"{n_indexed} chunks indexés avec {model_name}")

        results = []
        for i, qa in enumerate(qa_pairs):
            logger.info(f"Question {i+1}/{len(qa_pairs)}: {qa['question'][:60]}...")

            response = pipeline.query(qa["question"], n_chunks=n_chunks)
            metrics = self.evaluator.compute_metrics(
                question=qa["question"],
                answer=response.answer,
                expected_answer=qa["expected_answer"],
                chunks=response.retrieved_chunks
            )

            result = EvaluationResult(
                question=qa["question"],
                expected_answer=qa["expected_answer"],
                generated_answer=response.answer,
                embedding_model=model_name,
                metrics=metrics,
                retrieval_time_ms=response.retrieval_time_ms,
                generation_time_ms=response.generation_time_ms,
                n_chunks=response.n_chunks_retrieved
            )
            results.append(result)
            logger.info(f"  → Overall score: {metrics.overall_score:.3f}")

        avg_score = np.mean([r.metrics.overall_score for r in results])
        logger.info(f"Score moyen {model_name}: {avg_score:.3f}")
        return results, avg_score

    def compare_models(
        self,
        documents: List[Dict],
        qa_pairs: List[Dict],
        models: Optional[List[str]] = None
    ) -> ModelComparisonReport:
        """Compare tous les modèles d'embeddings disponibles."""
        from datetime import datetime

        if models is None:
            models = list(EMBEDDING_MODELS.keys())

        all_results = {}
        all_scores = {}

        for model_name in models:
            try:
                results, avg_score = self.evaluate_model(
                    model_name, documents, qa_pairs
                )
                all_results[model_name] = results
                all_scores[model_name] = round(avg_score, 4)
            except Exception as e:
                logger.error(f"Erreur avec {model_name}: {e}")
                all_scores[model_name] = 0.0

        best_model = max(all_scores, key=all_scores.get)

        report = ModelComparisonReport(
            models_evaluated=models,
            results_per_model=all_results,
            avg_scores=all_scores,
            best_model=best_model,
            evaluation_date=datetime.now().isoformat(),
            n_questions=len(qa_pairs)
        )

        self._print_report(report)
        return report

    def _print_report(self, report: ModelComparisonReport):
        """Affiche le rapport de comparaison."""
        print("\n" + "="*60)
        print("RAPPORT DE COMPARAISON DES MODÈLES D'EMBEDDINGS")
        print("="*60)
        print(f"Questions évaluées : {report.n_questions}")
        print(f"Date : {report.evaluation_date}\n")

        print(f"{'Modèle':<15} {'Score Global':>12} {'Classement':>10}")
        print("-"*40)

        sorted_models = sorted(report.avg_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, score) in enumerate(sorted_models, 1):
            marker = " 🏆" if model == report.best_model else ""
            print(f"{model:<15} {score:>12.4f} {rank:>8}{marker}")

        print(f"\n✅ Meilleur modèle : {report.best_model} (score: {report.avg_scores[report.best_model]:.4f})")
        print("="*60)
