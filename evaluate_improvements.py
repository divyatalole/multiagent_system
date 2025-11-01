#!/usr/bin/env python3
"""
Evaluate RAG Improvements
=========================

Compare retrieval performance before and after improvements:
- Query expansion
- Re-ranking
- Combined improvements
"""

import json
import logging
from pathlib import Path
from evaluation import RetrievalEvaluator, RetrievalResults
from rag_improvements import ImprovedRAGKnowledgeBase, QueryExpander, SimpleReRanker
from multi_agent_system_simple import RAGKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_improvements():
    """Evaluate all improvement strategies"""
    
    print("="*80)
    print("RAG RETRIEVAL IMPROVEMENT EVALUATION")
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
    print("\n1. Initializing base knowledge base...")
    kb = RAGKnowledgeBase()
    
    # Create improved version
    improved_kb = ImprovedRAGKnowledgeBase(kb)
    evaluator = RetrievalEvaluator()
    
    print("\n2. Evaluating improvements...\n")
    
    # Evaluate: Base (no improvements)
    print("="*80)
    print("BASELINE: No Improvements")
    print("="*80)
    base_results = evaluate_configuration(kb, queries, evaluator, "baseline")
    evaluator.print_results(base_results, "Baseline")
    
    # Evaluate: Query expansion only
    print("\n" + "="*80)
    print("IMPROVEMENT 1: Query Expansion Only")
    print("="*80)
    expansion_results = evaluate_configuration(
        kb, queries, evaluator, "expansion", use_expansion=True
    )
    evaluator.print_results(expansion_results, "Expansion")
    
    # Evaluate: Re-ranking only
    print("\n" + "="*80)
    print("IMPROVEMENT 2: Re-ranking Only")
    print("="*80)
    rerank_results = evaluate_configuration(
        kb, queries, evaluator, "rerank", rerank=True
    )
    evaluator.print_results(rerank_results, "Re-ranking")
    
    # Evaluate: Both improvements
    print("\n" + "="*80)
    print("IMPROVEMENT 3: Query Expansion + Re-ranking")
    print("="*80)
    combined_results = evaluate_configuration(
        kb, queries, evaluator, "combined", use_expansion=True, rerank=True
    )
    evaluator.print_results(combined_results, "Combined")
    
    # Print comparison
    print_comparison(base_results, expansion_results, rerank_results, combined_results)
    
    # Save results
    results = {
        'baseline': base_results.to_dict(),
        'expansion': expansion_results.to_dict(),
        'rerank': rerank_results.to_dict(),
        'combined': combined_results.to_dict()
    }
    
    with open('improvement_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to improvement_results.json")


def evaluate_configuration(
    kb: RAGKnowledgeBase,
    queries: list,
    evaluator: RetrievalEvaluator,
    config_name: str,
    use_expansion: bool = False,
    rerank: bool = False
) -> RetrievalResults:
    """
    Evaluate a specific configuration
    
    Args:
        kb: Knowledge base (base or improved)
        queries: Test queries
        evaluator: Evaluator instance
        config_name: Configuration name
        use_expansion: Use query expansion
        rerank: Use re-ranking
        
    Returns:
        RetrievalResults
    """
    total_queries = len(queries)
    
    # Accumulate metrics
    hit_rate_sum = 0.0
    precision_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    mrr_sum = 0.0
    ndcg_at_k = {1: 0.0, 3: 0.0, 5: 0.0}
    
    improved_kb = ImprovedRAGKnowledgeBase(kb)
    
    for query_data in queries:
        query = query_data['query']
        relevant_doc_ids = set(query_data.get('relevant_docs', []))
        relevance_scores = query_data.get('relevance_scores', {})
        role = query_data.get('role', 'General')
        
        # Retrieve documents based on configuration
        if use_expansion or rerank:
            # Use improved retrieval
            retrieved_docs = improved_kb.search_role_aware_with_expansion(
                query, role, max_results=5,
                use_expansion=use_expansion,
                rerank=rerank
            )
        else:
            # Use baseline retrieval
            retrieved_docs = kb.search_documents_for_role(query, role, max_results=5)
        
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
    
    # Average
    return RetrievalResults(
        hit_rate=hit_rate_sum / total_queries,
        precision_k={k: v / total_queries for k, v in precision_at_k.items()},
        recall_k={k: v / total_queries for k, v in recall_at_k.items()},
        mrr=mrr_sum / total_queries,
        ndcg_k={k: v / total_queries for k, v in ndcg_at_k.items()},
        total_queries=total_queries
    )


def print_comparison(*results):
    """Print comparison of all configurations"""
    print("\n" + "="*80)
    print("IMPROVEMENT COMPARISON")
    print("="*80)
    
    configs = ['Baseline', 'Expansion', 'Re-ranking', 'Combined']
    
    # Header
    print(f"\n{'Metric':<20} {'Baseline':<12} {'Expansion':<12} {'Re-rank':<12} {'Combined':<12} {'Best':<12}")
    print("-"*80)
    
    metrics = [
        ('Hit Rate', 'hit_rate'),
        ('MRR', 'mrr'),
        ('Precision@1', 'precision_k', 1),
        ('Precision@5', 'precision_k', 5),
        ('Recall@5', 'recall_k', 5),
        ('nDCG@5', 'ndcg_k', 5)
    ]
    
    for metric_name, metric_key, *subkey in metrics:
        row = f"{metric_name:<20}"
        values = []
        
        for result in results:
            if subkey:
                val = getattr(result, metric_key)[subkey[0]]
            else:
                val = getattr(result, metric_key)
            values.append(val)
            row += f"{val:<12.3f}"
        
        # Find best
        best_idx = values.index(max(values))
        best_config = configs[best_idx]
        row += f"{best_config:<12}"
        
        print(row)
    
    print("="*80)
    
    # Summary
    print("\nIMPROVEMENT SUMMARY:")
    print("-"*80)
    
    # Calculate improvements over baseline
    baseline = results[0]
    combined = results[3]
    
    p1_improvement = ((combined.precision_k[1] - baseline.precision_k[1]) / baseline.precision_k[1] * 100) if baseline.precision_k[1] > 0 else 0
    r5_improvement = ((combined.recall_k[5] - baseline.recall_k[5]) / baseline.recall_k[5] * 100) if baseline.recall_k[5] > 0 else 0
    mrr_improvement = ((combined.mrr - baseline.mrr) / baseline.mrr * 100) if baseline.mrr > 0 else 0
    
    print(f"Precision@1 improvement:  {p1_improvement:+.1f}%")
    print(f"Recall@5 improvement:     {r5_improvement:+.1f}%")
    print(f"MRR improvement:          {mrr_improvement:+.1f}%")
    print(f"nDCG@5 improvement:       {((combined.ndcg_k[5] - baseline.ndcg_k[5]) / baseline.ndcg_k[5] * 100):+.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    evaluate_improvements()


