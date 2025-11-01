#!/usr/bin/env python3
"""
Evaluate Cross-Encoder Re-ranking (Phase 2)
============================================

Benchmark the improvement from Cross-Encoder vs simple re-ranking.
"""

import json
import logging
from pathlib import Path
from evaluation import RetrievalEvaluator, RetrievalResults
from rag_improvements import ImprovedRAGKnowledgeBase, SimpleReRanker
from multi_agent_system_simple import RAGKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_cross_encoder():
    """Compare Cross-Encoder with simple re-ranking"""
    
    print("="*80)
    print("CROSS-ENCODER RE-RANKING EVALUATION (PHASE 2)")
    print("="*80)
    
    # Load test queries
    queries_file = Path("test_queries.json")
    if not queries_file.exists():
        print(f"Error: {queries_file} not found")
        return
    
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    logger.info(f"Loaded {len(queries)} test queries")
    
    # Initialize knowledge base
    print("\n1. Initializing knowledge base...")
    kb = RAGKnowledgeBase()
    improved_kb = ImprovedRAGKnowledgeBase(kb)
    evaluator = RetrievalEvaluator()
    
    print("\n2. Evaluating Cross-Encoder...\n")
    
    # Evaluate: Simple re-ranking (Phase 1 - baseline)
    print("="*80)
    print("PHASE 1: Simple Re-ranking (Bi-Encoder)")
    print("="*80)
    simple_results = evaluate_simple_reranking(kb, queries, evaluator)
    evaluator.print_results(simple_results, "Simple Re-ranking")
    
    # Evaluate: Cross-Encoder re-ranking (Phase 2)
    print("\n" + "="*80)
    print("PHASE 2: Cross-Encoder Re-ranking")
    print("="*80)
    
    try:
        cross_encoder_results = evaluate_cross_encoder_reranking(kb, queries, evaluator)
        evaluator.print_results(cross_encoder_results, "Cross-Encoder")
        
        # Comparison
        print_comparison(simple_results, cross_encoder_results)
        
        # Save results
        results = {
            'simple_reranking': simple_results.to_dict(),
            'cross_encoder': cross_encoder_results.to_dict()
        }
        
        with open('cross_encoder_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\nResults saved to cross_encoder_results.json")
        
    except Exception as e:
        logger.error(f"Cross-Encoder evaluation failed: {e}")
        logger.info("\nTo use Cross-Encoder, install: pip install sentence-transformers>=2.0.0")
        print("\nFalling back to simple re-ranking only.")


def evaluate_simple_reranking(kb, queries, evaluator):
    """Evaluate simple re-ranking performance"""
    improved_kb = ImprovedRAGKnowledgeBase(kb)
    
    total_queries = len(queries)
    hit_rate_sum = 0.0
    precision_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    mrr_sum = 0.0
    ndcg_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    
    for query_data in queries:
        query = query_data['query']
        relevant_doc_ids = set(query_data.get('relevant_docs', []))
        relevance_scores = query_data.get('relevance_scores', {})
        role = query_data.get('role', 'General')
        
        # Use simple re-ranking
        retrieved_docs = improved_kb.search_role_aware_with_expansion(
            query, role, max_results=5, use_expansion=False, rerank=True
        )
        
        # Calculate metrics
        hit_rate_sum += evaluator.calculate_hit_rate(retrieved_docs, relevant_doc_ids)
        mrr_sum += evaluator.calculate_mrr(retrieved_docs, relevant_doc_ids)
        
        for k in [1, 3, 5]:
            precision_at_k[k] += evaluator.calculate_precision_at_k(
                retrieved_docs, relevant_doc_ids, k
            )
            recall_at_k[k] += evaluator.calculate_recall_at_k(
                retrieved_docs, relevant_doc_ids, k
            )
            ndcg_at_k[k] += evaluator.calculate_ndcg_at_k(
                retrieved_docs, relevant_doc_ids, relevance_scores, k
            )
    
    return RetrievalResults(
        hit_rate=hit_rate_sum / total_queries,
        precision_k={k: v / total_queries for k, v in precision_at_k.items()},
        recall_k={k: v / total_queries for k, v in recall_at_k.items()},
        mrr=mrr_sum / total_queries,
        ndcg_k={k: v / total_queries for k, v in ndcg_at_k.items()},
        total_queries=total_queries
    )


def evaluate_cross_encoder_reranking(kb, queries, evaluator):
    """Evaluate Cross-Encoder re-ranking performance"""
    improved_kb = ImprovedRAGKnowledgeBase(kb)
    
    total_queries = len(queries)
    hit_rate_sum = 0.0
    precision_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    mrr_sum = 0.0
    ndcg_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    
    for i, query_data in enumerate(queries, 1):
        query = query_data['query']
        relevant_doc_ids = set(query_data.get('relevant_docs', []))
        relevance_scores = query_data.get('relevance_scores', {})
        role = query_data.get('role', 'General')
        
        # Use Cross-Encoder re-ranking
        try:
            retrieved_docs = improved_kb.search_with_cross_encoder_reranking(
                query, role, max_results=5, retrieval_candidates=30
            )
        except Exception as e:
            logger.warning(f"Query {i} failed with Cross-Encoder: {e}, using simple re-ranking")
            retrieved_docs = improved_kb.search_role_aware_with_expansion(
                query, role, max_results=5, use_expansion=False, rerank=True
            )
        
        # Calculate metrics
        hit_rate_sum += evaluator.calculate_hit_rate(retrieved_docs, relevant_doc_ids)
        mrr_sum += evaluator.calculate_mrr(retrieved_docs, relevant_doc_ids)
        
        for k in [1, 3, 5]:
            precision_at_k[k] += evaluator.calculate_precision_at_k(
                retrieved_docs, relevant_doc_ids, k
            )
            recall_at_k[k] += evaluator.calculate_recall_at_k(
                retrieved_docs, relevant_doc_ids, k
            )
            ndcg_at_k[k] += evaluator.calculate_ndcg_at_k(
                retrieved_docs, relevant_doc_ids, relevance_scores, k
            )
    
    return RetrievalResults(
        hit_rate=hit_rate_sum / total_queries,
        precision_k={k: v / total_queries for k, v in precision_at_k.items()},
        recall_k={k: v / total_queries for k, v in recall_at_k.items()},
        mrr=mrr_sum / total_queries,
        ndcg_k={k: v / total_queries for k, v in ndcg_at_k.items()},
        total_queries=total_queries
    )


def print_comparison(simple_results, ce_results):
    """Print side-by-side comparison"""
    print("\n" + "="*80)
    print("PHASE 1 vs PHASE 2 COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Simple Re-rank':<18} {'Cross-Encoder':<18} {'Improvement':<15}")
    print("-"*76)
    
    metrics = [
        ('Hit Rate', 'hit_rate'),
        ('MRR', 'mrr'),
        ('Precision@1', 'precision_k', 1),
        ('Precision@5', 'precision_k', 5),
        ('Recall@5', 'recall_k', 5),
        ('nDCG@5', 'ndcg_k', 5)
    ]
    
    for metric_name, metric_key, *subkey in metrics:
        if subkey:
            simple_val = getattr(simple_results, metric_key)[subkey[0]]
            ce_val = getattr(ce_results, metric_key)[subkey[0]]
        else:
            simple_val = getattr(simple_results, metric_key)
            ce_val = getattr(ce_results, metric_key)
        
        improvement = ((ce_val - simple_val) / simple_val * 100) if simple_val > 0 else 0
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        
        print(f"{metric_name:<25} {simple_val:<18.3f} {ce_val:<18.3f} {improvement_str:<15}")
    
    print("="*80)
    
    # Summary
    print("\nSUMMARY:")
    print("-"*80)
    
    p1_imp = ((ce_results.precision_k[1] - simple_results.precision_k[1]) / simple_results.precision_k[1] * 100)
    mrr_imp = ((ce_results.mrr - simple_results.mrr) / simple_results.mrr * 100)
    ndcg_imp = ((ce_results.ndcg_k[5] - simple_results.ndcg_k[5]) / simple_results.ndcg_k[5] * 100)
    
    print(f"Precision@1 improvement:  {p1_imp:+.1f}%")
    print(f"MRR improvement:          {mrr_imp:+.1f}%")
    print(f"nDCG@5 improvement:       {ndcg_imp:+.1f}%")
    
    if ndcg_imp > 0:
        print("\n[SUCCESS] Cross-Encoder provides more accurate document ranking!")
    else:
        print("\n[INFO] Cross-Encoder shows mixed results - may need tuning")
    print("="*80)


if __name__ == "__main__":
    evaluate_cross_encoder()

