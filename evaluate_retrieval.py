#!/usr/bin/env python3
"""
Retrieval Quality Evaluation Script
====================================

Comprehensive evaluation of RAG retrieval quality with standardized metrics.
"""

import json
import argparse
from pathlib import Path
import logging
from evaluation import RetrievalEvaluator, RetrievalResults
from multi_agent_system_simple import RAGKnowledgeBase, RAGMultiAgentSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_queries(file_path: str) -> list:
    """Load test queries with ground truth"""
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries


def evaluate_general_retrieval(kb: RAGKnowledgeBase, queries: list) -> RetrievalResults:
    """Evaluate general retrieval (non-role-aware)"""
    logger.info("Evaluating general retrieval...")
    evaluator = RetrievalEvaluator()
    
    # Convert queries to format expected by evaluator
    test_queries = [
        {
            'query': q['query'],
            'relevant_docs': q['relevant_docs'],
            'relevance_scores': q.get('relevance_scores', {})
        }
        for q in queries
    ]
    
    results = evaluator.evaluate_batch(test_queries, kb)
    return results


def evaluate_role_aware_retrieval(kb: RAGKnowledgeBase, queries: list, role: str) -> RetrievalResults:
    """Evaluate role-aware retrieval for specific role"""
    logger.info(f"Evaluating role-aware retrieval for {role}...")
    evaluator = RetrievalEvaluator()
    
    # Filter queries for the specific role
    role_queries = [q for q in queries if q.get('role') == role]
    
    if not role_queries:
        logger.warning(f"No queries found for role: {role}")
        return RetrievalResults(0, {}, {}, 0, {}, 0)
    
    test_queries = [
        {
            'query': q['query'],
            'relevant_docs': q['relevant_docs'],
            'relevance_scores': q.get('relevance_scores', {})
        }
        for q in role_queries
    ]
    
    results = evaluator.evaluate_role_aware(test_queries, kb, role)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate RAG retrieval quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate general retrieval
  python evaluate_retrieval.py --queries test_queries.json
  
  # Evaluate all roles
  python evaluate_retrieval.py --queries test_queries.json --all-roles
  
  # Evaluate specific role
  python evaluate_retrieval.py --queries test_queries.json --role Investor
  
  # Save results to JSON
  python evaluate_retrieval.py --queries test_queries.json --output results.json
        """
    )
    
    parser.add_argument(
        '--queries',
        type=str,
        default='test_queries.json',
        help='Path to test queries JSON file'
    )
    
    parser.add_argument(
        '--role',
        type=str,
        choices=['Investor', 'Researcher', 'User'],
        help='Evaluate specific role'
    )
    
    parser.add_argument(
        '--all-roles',
        action='store_true',
        help='Evaluate all three roles separately'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--skip-general',
        action='store_true',
        help='Skip general retrieval evaluation'
    )
    
    args = parser.parse_args()
    
    # Load test queries
    logger.info(f"Loading test queries from {args.queries}")
    try:
        queries = load_test_queries(args.queries)
        logger.info(f"Loaded {len(queries)} test queries")
    except FileNotFoundError:
        logger.error(f"Test queries file not found: {args.queries}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in queries file: {e}")
        return 1
    
    # Initialize knowledge base
    logger.info("Initializing RAG knowledge base...")
    try:
        kb = RAGKnowledgeBase()
        logger.info("Knowledge base initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        return 1
    
    all_results = {}
    
    # Evaluate general retrieval
    if not args.skip_general:
        logger.info("\n" + "="*70)
        logger.info("GENERAL RETRIEVAL EVALUATION")
        logger.info("="*70)
        general_results = evaluate_general_retrieval(kb, queries)
        evaluator = RetrievalEvaluator()
        evaluator.print_results(general_results, "General")
        all_results['general'] = general_results.to_dict()
    
    # Evaluate by role
    if args.all_roles:
        roles = ['Investor', 'Researcher', 'User']
        logger.info("\n" + "="*70)
        logger.info("ROLE-AWARE RETRIEVAL EVALUATION")
        logger.info("="*70)
        
        for role in roles:
            role_results = evaluate_role_aware_retrieval(kb, queries, role)
            evaluator = RetrievalEvaluator()
            evaluator.print_results(role_results, role)
            all_results[f'role_{role.lower()}'] = role_results.to_dict()
    
    elif args.role:
        role_results = evaluate_role_aware_retrieval(kb, queries, args.role)
        evaluator = RetrievalEvaluator()
        evaluator.print_results(role_results, args.role)
        all_results[f'role_{args.role.lower()}'] = role_results.to_dict()
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'evaluation_summary': all_results,
                'total_test_queries': len(queries)
            }, f, indent=2)
        
        logger.info(f"\nResults saved to {output_path}")
    
    # Print summary comparison if evaluating multiple modes
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print(f"{'Metric':<25} {'General':<12} {'Investor':<12} {'Researcher':<12} {'User':<12}")
        print("-"*70)
        
        metrics = [
            ('Hit Rate', 'hit_rate'),
            ('MRR', 'mrr'),
            ('Precision@5', 'precision_k', 5),
            ('Recall@5', 'recall_k', 5),
            ('nDCG@5', 'ndcg_k', 5)
        ]
        
        for metric_name, metric_key, *args_subkey in metrics:
            row = f"{metric_name:<25}"
            
            # General
            if 'general' in all_results:
                if args_subkey:
                    val = all_results['general'][metric_key][args_subkey[0]]
                else:
                    val = all_results['general'][metric_key]
                row += f"{val:<12.3f}"
            else:
                row += f"{'N/A':<12}"
            
            # Roles
            for role in ['investor', 'researcher', 'user']:
                key = f'role_{role}'
                if key in all_results:
                    if args_subkey:
                        val = all_results[key][metric_key][args_subkey[0]]
                    else:
                        val = all_results[key][metric_key]
                    row += f"{val:<12.3f}"
                else:
                    row += f"{'N/A':<12}"
            
            print(row)
        
        print("="*70)
    
    logger.info("\n✅ Evaluation complete!")
    return 0


if __name__ == "__main__":
    exit(main())

