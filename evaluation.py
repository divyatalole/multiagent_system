"""
Retrieval Quality Evaluation Module
====================================

Comprehensive evaluation metrics for RAG retrieval quality:
- Hit Rate
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@k)
"""

import numpy as np
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResults:
    """Container for retrieval evaluation results"""
    hit_rate: float
    precision_k: Dict[int, float]  # precision@1, precision@3, precision@5
    recall_k: Dict[int, float]  # recall@1, recall@3, recall@5
    mrr: float
    ndcg_k: Dict[int, float]  # nDCG@1, nDCG@3, nDCG@5
    total_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization"""
        return {
            'hit_rate': self.hit_rate,
            'precision_k': self.precision_k,
            'recall_k': self.recall_k,
            'mrr': self.mrr,
            'ndcg_k': self.ndcg_k,
            'total_queries': self.total_queries
        }


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard information retrieval metrics
    """
    
    def __init__(self):
        self.relevance_threshold = 0.5  # Minimum similarity to be considered relevant
        
    def calculate_hit_rate(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        relevant_doc_ids: Set[str]
    ) -> float:
        """
        Hit Rate: Did the retriever find any relevant document in its top-K results?
        
        Args:
            retrieved_docs: List of retrieved documents with 'source' field
            relevant_doc_ids: Set of relevant document IDs (sources)
            
        Returns:
            Hit rate (0.0 to 1.0) - fraction of queries with at least one relevant doc
        """
        if not retrieved_docs:
            return 0.0
        
        # Check if any retrieved document is relevant
        retrieved_sources = {doc.get('source', '') for doc in retrieved_docs}
        has_relevant = len(retrieved_sources & relevant_doc_ids) > 0
        
        return 1.0 if has_relevant else 0.0
    
    def calculate_precision_at_k(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """
        Precision@k: Of the top-K documents retrieved, what percentage were relevant?
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Precision@k (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
            
        top_k_docs = retrieved_docs[:k]
        if not top_k_docs:
            return 0.0
        
        retrieved_sources = {doc.get('source', '') for doc in top_k_docs}
        relevant_retrieved = retrieved_sources & relevant_doc_ids
        
        return len(relevant_retrieved) / len(top_k_docs)
    
    def calculate_recall_at_k(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """
        Recall@k: Of all possible relevant documents, what percentage did the retriever find?
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: Set of all relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Recall@k (0.0 to 1.0)
        """
        if not relevant_doc_ids:
            return 1.0  # No relevant docs means perfect recall
        
        top_k_docs = retrieved_docs[:k]
        if not top_k_docs:
            return 0.0
        
        retrieved_sources = {doc.get('source', '') for doc in top_k_docs}
        relevant_retrieved = retrieved_sources & relevant_doc_ids
        
        return len(relevant_retrieved) / len(relevant_doc_ids)
    
    def calculate_mrr(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        relevant_doc_ids: Set[str]
    ) -> float:
        """
        Mean Reciprocal Rank (MRR): How high up in the list was the first relevant document?
        Great for "fact-finding" tasks where one good answer is enough.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: Set of relevant document IDs
            
        Returns:
            MRR (0.0 to 1.0), where 1.0 means first result is relevant
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0
        
        # Find the rank of the first relevant document (1-indexed)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc.get('source', '') in relevant_doc_ids:
                return 1.0 / rank
        
        return 0.0  # No relevant document found
    
    def calculate_ndcg_at_k(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        relevant_doc_ids: Set[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain (nDCG@k): The "gold standard" metric.
        Rewards for retrieving highly relevant documents and for ranking them higher.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: Set of relevant document IDs
            relevance_scores: Dict mapping doc_id to relevance score (0=irrelevant, 1=highly relevant)
            k: Number of top documents to consider
            
        Returns:
            nDCG@k (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        if not top_k_docs:
            return 0.0
        
        # Calculate DCG@k
        dcg = 0.0
        for rank, doc in enumerate(top_k_docs, start=1):
            doc_id = doc.get('source', '')
            if doc_id in relevant_doc_ids:
                relevance = relevance_scores.get(doc_id, 1.0)  # Default to 1.0 if not specified
                dcg += relevance / np.log2(rank + 1)
        
        # Calculate Ideal DCG@k (IDCG@k)
        # Sort relevance scores in descending order
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
        
        # Normalize
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_batch(
        self,
        queries: List[Dict[str, Any]],
        knowledge_base: Any  # RAGKnowledgeBase instance
    ) -> RetrievalResults:
        """
        Evaluate retrieval quality on a batch of queries with ground truth
        
        Args:
            queries: List of query dicts with format:
                {
                    'query': str,
                    'relevant_docs': List[str],  # List of relevant document sources
                    'relevance_scores': Dict[str, float]  # Optional: doc_id -> relevance score
                }
            knowledge_base: RAGKnowledgeBase instance
            
        Returns:
            RetrievalResults with all metrics
        """
        total_queries = len(queries)
        
        # Accumulate metrics
        hit_rate_sum = 0.0
        precision_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
        recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
        mrr_sum = 0.0
        ndcg_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
        
        for query_data in queries:
            query = query_data['query']
            relevant_doc_ids = set(query_data.get('relevant_docs', []))
            relevance_scores = query_data.get('relevance_scores', {})
            
            # Retrieve documents
            retrieved_docs = knowledge_base.search_documents(query, max_results=5)
            
            # Calculate metrics
            hit_rate_sum += self.calculate_hit_rate(retrieved_docs, relevant_doc_ids)
            mrr_sum += self.calculate_mrr(retrieved_docs, relevant_doc_ids)
            
            for k in [1, 3, 5]:
                precision_at_k[k] += self.calculate_precision_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                recall_at_k[k] += self.calculate_recall_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                ndcg_at_k[k] += self.calculate_ndcg_at_k(
                    retrieved_docs, relevant_doc_ids, relevance_scores, k
                )
        
        # Average over all queries
        avg_hit_rate = hit_rate_sum / total_queries if total_queries > 0 else 0.0
        avg_mrr = mrr_sum / total_queries if total_queries > 0 else 0.0
        
        avg_precision_at_k = {
            k: precision_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in [1, 3, 5]
        }
        
        avg_recall_at_k = {
            k: recall_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in [1, 3, 5]
        }
        
        avg_ndcg_at_k = {
            k: ndcg_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in [1, 3, 5]
        }
        
        return RetrievalResults(
            hit_rate=avg_hit_rate,
            precision_k=avg_precision_at_k,
            recall_k=avg_recall_at_k,
            mrr=avg_mrr,
            ndcg_k=avg_ndcg_at_k,
            total_queries=total_queries
        )
    
    def evaluate_role_aware(
        self,
        queries: List[Dict[str, Any]],
        knowledge_base: Any,
        role: str
    ) -> RetrievalResults:
        """
        Evaluate role-aware retrieval quality
        
        Args:
            queries: List of query dicts with ground truth
            knowledge_base: RAGKnowledgeBase instance
            role: Agent role (Investor, Researcher, User)
            
        Returns:
            RetrievalResults with all metrics
        """
        total_queries = len(queries)
        
        # Accumulate metrics
        hit_rate_sum = 0.0
        precision_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
        recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
        mrr_sum = 0.0
        ndcg_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
        
        for query_data in queries:
            query = query_data['query']
            relevant_doc_ids = set(query_data.get('relevant_docs', []))
            relevance_scores = query_data.get('relevance_scores', {})
            
            # Retrieve documents with role awareness
            retrieved_docs = knowledge_base.search_documents_for_role(
                query, role, max_results=5
            )
            
            # Calculate metrics
            hit_rate_sum += self.calculate_hit_rate(retrieved_docs, relevant_doc_ids)
            mrr_sum += self.calculate_mrr(retrieved_docs, relevant_doc_ids)
            
            for k in [1, 3, 5]:
                precision_at_k[k] += self.calculate_precision_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                recall_at_k[k] += self.calculate_recall_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                ndcg_at_k[k] += self.calculate_ndcg_at_k(
                    retrieved_docs, relevant_doc_ids, relevance_scores, k
                )
        
        # Average over all queries
        avg_hit_rate = hit_rate_sum / total_queries if total_queries > 0 else 0.0
        avg_mrr = mrr_sum / total_queries if total_queries > 0 else 0.0
        
        avg_precision_at_k = {
            k: precision_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in [1, 3, 5]
        }
        
        avg_recall_at_k = {
            k: recall_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in [1, 3, 5]
        }
        
        avg_ndcg_at_k = {
            k: ndcg_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in [1, 3, 5]
        }
        
        return RetrievalResults(
            hit_rate=avg_hit_rate,
            precision_k=avg_precision_at_k,
            recall_k=avg_recall_at_k,
            mrr=avg_mrr,
            ndcg_k=avg_ndcg_at_k,
            total_queries=total_queries
        )
    
    def print_results(self, results: RetrievalResults, role: str = "General"):
        """
        Print formatted evaluation results
        
        Args:
            results: RetrievalResults object
            role: Role name for context
        """
        print(f"\n{'='*70}")
        print(f"RETRIEVAL QUALITY EVALUATION - {role}")
        print(f"{'='*70}")
        print(f"\nTotal Queries Evaluated: {results.total_queries}")
        print(f"\n[METRICS SUMMARY]")
        print(f"{'-'*70}")
        print(f"Hit Rate:              {results.hit_rate:.3f} ({results.hit_rate*100:.1f}%)")
        print(f"Mean Reciprocal Rank:  {results.mrr:.3f}")
        print(f"\nPrecision@k:")
        print(f"  @1:  {results.precision_k[1]:.3f} ({results.precision_k[1]*100:.1f}%)")
        print(f"  @3:  {results.precision_k[3]:.3f} ({results.precision_k[3]*100:.1f}%)")
        print(f"  @5:  {results.precision_k[5]:.3f} ({results.precision_k[5]*100:.1f}%)")
        print(f"\nRecall@k:")
        print(f"  @1:  {results.recall_k[1]:.3f} ({results.recall_k[1]*100:.1f}%)")
        print(f"  @3:  {results.recall_k[3]:.3f} ({results.recall_k[3]*100:.1f}%)")
        print(f"  @5:  {results.recall_k[5]:.3f} ({results.recall_k[5]*100:.1f}%)")
        print(f"\nnDCG@k (Gold Standard):")
        print(f"  @1:  {results.ndcg_k[1]:.3f}")
        print(f"  @3:  {results.ndcg_k[3]:.3f}")
        print(f"  @5:  {results.ndcg_k[5]:.3f}")
        print(f"{'='*70}\n")
    
    def save_results(self, results: RetrievalResults, output_path: str, role: str = "General"):
        """
        Save evaluation results to JSON file
        
        Args:
            results: RetrievalResults object
            output_path: Path to save JSON file
            role: Role name for context
        """
        output_data = {
            'role': role,
            'metrics': results.to_dict()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

