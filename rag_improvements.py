"""
RAG Retrieval Improvements
==========================

This module implements advanced retrieval improvements:
1. Dynamic chunking optimization
2. Query expansion for better recall
3. Cross-encoder re-ranker for precision
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for optimized chunking"""
    small_chunk_size: int = 500      # For User queries (specific info)
    medium_chunk_size: int = 1000    # For Investor queries (balanced)
    large_chunk_size: int = 2000     # For Researcher queries (context-heavy)
    chunk_overlap: int = 200


class QueryExpander:
    """
    Expands simple user queries into more detailed queries for better retrieval
    """
    
    def __init__(self):
        self.expansion_templates = {
            "investor": [
                "Focus on: {query}. Consider ROI, revenue, margins, market size (TAM/SAM), "
                "funding stages, valuation metrics, competitive landscape, unit economics, "
                "and investment risk factors.",
                "{query}. Include: financial viability, market opportunity, competitive advantage, "
                "revenue model, growth potential, and exit strategy considerations.",
                "{query}. Analyze: investment potential, market trends, competition analysis, "
                "financial projections, and investor due diligence factors."
            ],
            "user": [
                "{query}. Consider: user experience, adoption patterns, onboarding process, "
                "retention strategies, usability, pricing models, and customer feedback.",
                "{query}. Focus on: practical implementation, user journey, feature adoption, "
                "support systems, ease of use, and user satisfaction metrics.",
                "{query}. Include: user needs, market fit, engagement strategies, accessibility, "
                "design principles, and user acquisition approaches."
            ],
            "researcher": [
                "{query}. Consider: technical feasibility, state-of-the-art methods, benchmarks, "
                "datasets, architectures, limitations, innovation opportunities, and research gaps.",
                "{query}. Focus on: technical approaches, algorithm design, evaluation metrics, "
                "implementation details, and technological trends."
            ]
        }
    
    def expand_query(self, query: str, role: str) -> str:
        """
        Expand a query with role-specific context
        
        Args:
            query: Original user query
            role: Agent role (Investor, Researcher, User)
            
        Returns:
            Expanded query string
        """
        role_lower = role.lower()
        
        # Get expansion template for role
        if role_lower not in self.expansion_templates:
            logger.warning(f"No expansion template for role: {role}")
            return query
        
        templates = self.expansion_templates[role_lower]
        
        # Use the first template (can be randomized or selected based on query)
        expanded = templates[0].format(query=query)
        
        logger.debug(f"Query expanded from '{query}' to '{expanded[:100]}...'")
        return expanded
    
    def expand_with_keywords(self, query: str, role: str) -> str:
        """
        Simple keyword-based expansion for fast retrieval
        
        Args:
            query: Original query
            role: Agent role
            
        Returns:
            Query with role-specific keywords
        """
        keyword_maps = {
            "investor": [
                "ROI", "revenue", "profit", "margin", "TAM", "SAM", "market size",
                "valuation", "funding", "investment", "risk", "competition", "growth",
                "unit economics", "payback period", "cagr"
            ],
            "user": [
                "user experience", "UX", "adoption", "onboarding", "retention",
                "engagement", "usability", "design", "pricing", "support",
                "feedback", "satisfaction", "accessibility"
            ],
            "researcher": [
                "state of the art", "benchmark", "dataset", "algorithm", "accuracy",
                "method", "architecture", "innovation", "technical", "feasibility"
            ]
        }
        
        role_lower = role.lower()
        if role_lower not in keyword_maps:
            return query
        
        keywords = keyword_maps[role_lower]
        
        # Add top 3-5 keywords that aren't already in the query
        query_lower = query.lower()
        missing_keywords = [kw for kw in keywords if kw.lower() not in query_lower][:5]
        
        if missing_keywords:
            expanded = f"{query} {' '.join(missing_keywords)}"
            logger.debug(f"Added keywords to query: {expanded[:150]}...")
            return expanded
        
        return query


class SimpleReRanker:
    """
    Lightweight re-ranker using embedding similarity and keyword matching
    
    This is a Bi-Encoder approach: queries and documents are encoded separately,
    then compared. Fast but less accurate than Cross-Encoder.
    """
    
    def __init__(self):
        self.relevance_boost = 0.3  # Weight for keyword matching
        self.cross_encoder = None  # Optional Cross-Encoder for advanced re-ranking
    
    def rerank(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using combined semantic and keyword signals
        
        Args:
            query: Original query
            retrieved_docs: List of retrieved documents with 'content' and 'relevance'
            top_k: Number of top documents to return
            
        Returns:
            Re-ranked list of documents
        """
        if not retrieved_docs:
            return []
        
        # Extract query keywords
        query_words = set(query.lower().split())
        
        # Score each document
        scored_docs = []
        for doc in retrieved_docs:
            semantic_score = doc.get('relevance', 0.0)
            content = doc.get('content', '').lower()
            
            # Keyword matching score
            content_words = set(content.split())
            common_words = query_words & content_words
            keyword_score = len(common_words) / max(len(query_words), 1)
            
            # Combine scores
            final_score = semantic_score + (self.relevance_boost * keyword_score)
            
            scored_docs.append({
                **doc,
                '_rerank_score': final_score,
                '_keyword_score': keyword_score
            })
        
        # Sort by re-rank score
        scored_docs.sort(key=lambda x: x['_rerank_score'], reverse=True)
        
        # Remove internal scoring fields
        for doc in scored_docs:
            doc.pop('_rerank_score', None)
            doc.pop('_keyword_score', None)
        
        logger.debug(f"Re-ranked {len(retrieved_docs)} documents, returning top {top_k}")
        return scored_docs[:top_k]
    
    def rerank_with_semantic_similarity(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        embedding_model: Any,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank using query-document embedding similarity
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            embedding_model: Sentence transformer model
            top_k: Number of documents to return
            
        Returns:
            Re-ranked documents
        """
        if not retrieved_docs:
            return []
        
        # Get query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Score each document
        for doc in retrieved_docs:
            content = doc.get('content', '')
            doc_embedding = embedding_model.encode([content], normalize_embeddings=True)[0]
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            
            # Combine with original relevance
            original_score = doc.get('relevance', 0.0)
            final_score = 0.7 * original_score + 0.3 * similarity
            
            doc['_rerank_score'] = final_score
        
        # Sort and return top-k
        retrieved_docs.sort(key=lambda x: x.get('_rerank_score', 0.0), reverse=True)
        
        # Clean up
        for doc in retrieved_docs:
            doc.pop('_rerank_score', None)
        
        return retrieved_docs[:top_k]
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k: int = 5,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Re-rank using Cross-Encoder for maximum accuracy
        
        Cross-Encoder processes (query, document) pairs together, allowing it
        to model direct interactions between query and document words.
        MUCH more accurate than Bi-Encoder, but slower.
        
        Use this for re-ranking top 25-50 candidates from fast retrieval.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents with 'content'
            cross_encoder_model: HuggingFace model identifier
            top_k: Number of top documents to return
            batch_size: Batch size for processing
            
        Returns:
            Re-ranked documents ordered by relevance
        """
        if not retrieved_docs:
            return []
        
        try:
            from sentence_transformers import CrossEncoder as SECrossEncoder
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            # Fallback to simple re-ranking
            return self.rerank(query, retrieved_docs, top_k)
        
        # Initialize Cross-Encoder (lazy load to avoid dependency issues)
        if self.cross_encoder is None:
            logger.info(f"Loading Cross-Encoder model: {cross_encoder_model}")
            self.cross_encoder = SECrossEncoder(cross_encoder_model, max_length=512)
        
        # Prepare query-document pairs
        pairs = [[query, doc.get('content', '')[:512]] for doc in retrieved_docs]
        
        # Get relevance scores from Cross-Encoder
        logger.debug(f"Re-ranking {len(pairs)} documents with Cross-Encoder")
        try:
            scores = self.cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Cross-Encoder prediction failed: {e}, falling back to simple re-ranking")
            return self.rerank(query, retrieved_docs, top_k)
        
        # Normalize Cross-Encoder scores to [0, 1]
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score > min_score:
            scores_normalized = (scores_array - min_score) / (max_score - min_score)
        else:
            scores_normalized = np.ones_like(scores_array) * 0.5  # All equal if no variance
        
        # Combine Cross-Encoder scores with original semantic scores
        for doc, ce_score, ce_normalized in zip(retrieved_docs, scores, scores_normalized):
            original_score = doc.get('relevance', 0.0)
            
            # Weighted combination: 70% Cross-Encoder, 30% original
            final_score = 0.7 * float(ce_normalized) + 0.3 * original_score
            
            doc['_rerank_score'] = final_score
            doc['_ce_score'] = float(ce_normalized)
        
        # Sort by final score
        retrieved_docs.sort(key=lambda x: x.get('_rerank_score', 0.0), reverse=True)
        
        # Clean up internal scores (keep for debugging if needed)
        for doc in retrieved_docs:
            doc.pop('_rerank_score', None)
            doc.pop('_ce_score', None)
        
        logger.debug(f"Cross-Encoder re-ranking complete, returning top {top_k}")
        return retrieved_docs[:top_k]


class ImprovedRAGKnowledgeBase:
    """
    Enhanced RAG knowledge base with optimized chunking and re-ranking
    """
    
    def __init__(self, base_kb: Any):
        """
        Wrap existing RAGKnowledgeBase with improvements
        
        Args:
            base_kb: Existing RAGKnowledgeBase instance
        """
        self.base_kb = base_kb
        self.query_expander = QueryExpander()
        self.re_ranker = SimpleReRanker()
        self.chunking_config = ChunkingConfig()
    
    def search_with_reranking(
        self, 
        query: str, 
        max_results: int = 5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with optional re-ranking
        
        Args:
            query: Search query
            max_results: Number of results to return
            rerank: Whether to apply re-ranking
            
        Returns:
            List of retrieved documents
        """
        # Retrieve more documents initially if re-ranking
        initial_k = max_results * 3 if rerank else max_results
        retrieved = self.base_kb.search_documents(query, max_results=initial_k)
        
        if not rerank or not retrieved:
            return retrieved[:max_results]
        
        # Apply re-ranking
        reranked = self.re_ranker.rerank(query, retrieved, top_k=max_results)
        return reranked
    
    def search_role_aware_with_expansion(
        self, 
        query: str, 
        role: str, 
        max_results: int = 5,
        use_expansion: bool = True,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Role-aware search with query expansion and re-ranking
        
        Args:
            query: Original query
            role: Agent role
            max_results: Number of results
            use_expansion: Whether to expand query
            rerank: Whether to apply re-ranking
            
        Returns:
            Retrieved and re-ranked documents
        """
        # Expand query if requested
        if use_expansion:
            expanded_query = self.query_expander.expand_with_keywords(query, role)
        else:
            expanded_query = query
        
        # Retrieve more documents if re-ranking
        initial_k = max_results * 3 if rerank else max_results
        
        # Use base role-aware search
        retrieved = self.base_kb.search_documents_for_role(
            expanded_query, role, max_results=initial_k
        )
        
        if not rerank or not retrieved:
            return retrieved[:max_results]
        
        # Apply re-ranking
        reranked = self.re_ranker.rerank(query, retrieved, top_k=max_results)
        return reranked
    
    def search_with_semantic_reranking(
        self,
        query: str,
        role: str,
        max_results: int = 5,
        use_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with semantic re-ranking using embedding similarity (Bi-Encoder)
        
        Args:
            query: Search query
            role: Agent role
            max_results: Number of results
            use_expansion: Whether to expand query
            
        Returns:
            Retrieved and semantically re-ranked documents
        """
        # Expand query
        if use_expansion:
            expanded_query = self.query_expander.expand_with_keywords(query, role)
        else:
            expanded_query = query
        
        # Retrieve with base method
        retrieved = self.base_kb.search_documents_for_role(
            expanded_query, role, max_results=max_results * 3
        )
        
        if not retrieved:
            return []
        
        # Apply semantic re-ranking
        reranked = self.re_ranker.rerank_with_semantic_similarity(
            query, retrieved, self.base_kb.embedding_model, top_k=max_results
        )
        
        return reranked
    
    def search_with_cross_encoder_reranking(
        self,
        query: str,
        role: str,
        max_results: int = 5,
        use_expansion: bool = False,
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        retrieval_candidates: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Advanced two-stage retrieval with Cross-Encoder re-ranking (Phase 2)
        
        Stage 1: Fast retrieval with Bi-Encoder (retrieve top 25-50 candidates)
        Stage 2: Accurate re-ranking with Cross-Encoder (select top 5-10)
        
        This is the recommended approach for production: fast initial retrieval,
        then accurate re-ranking of top candidates.
        
        Args:
            query: Original search query
            role: Agent role for role-aware retrieval
            max_results: Final number of results to return
            use_expansion: Whether to expand query for retrieval (not recommended)
            cross_encoder_model: Cross-Encoder model identifier
            retrieval_candidates: Number of candidates to retrieve before re-ranking
            
        Returns:
            Retrieved and Cross-Encoder re-ranked documents
        """
        # Stage 1: Fast retrieval (do NOT expand here - hurts precision)
        expanded_query = query  # Use original query for retrieval
        
        # Retrieve larger candidate set
        retrieved = self.base_kb.search_documents_for_role(
            expanded_query, role, max_results=retrieval_candidates
        )
        
        if not retrieved:
            return []
        
        # Stage 2: Accurate re-ranking with Cross-Encoder
        logger.info(f"Re-ranking {len(retrieved)} candidates with Cross-Encoder")
        reranked = self.re_ranker.rerank_with_cross_encoder(
            query, retrieved, cross_encoder_model, top_k=max_results
        )
        
        return reranked


def apply_improvements_to_agent(agent: Any, improvements: ImprovedRAGKnowledgeBase) -> Any:
    """
    Apply improvements to an RAG agent
    
    Args:
        agent: RAGAgent instance
        improvements: ImprovedRAGKnowledgeBase instance
        
    Returns:
        Agent with improved retrieval
    """
    # Monkey-patch the agent to use improved retrieval
    original_analyze = agent.analyze_topic
    
    def improved_analyze(topic: str) -> Dict[str, Any]:
        # Get role
        role = agent.role
        
        # Use improved retrieval
        relevant_docs = improvements.search_role_aware_with_expansion(
            topic, role, max_results=5, use_expansion=True, rerank=True
        )
        
        # Continue with original analysis logic
        analysis = {
            'agent': agent.name,
            'role': role,
            'topic': topic,
            'relevant_documents': len(relevant_docs),
            'insights': [],
            'recommendations': [],
            'llm_analysis': None,
            'document_previews': [
                {
                    'source': doc['source'],
                    'type': doc['type'],
                    'preview': doc['preview'],
                    'relevance': doc['relevance']
                }
                for doc in relevant_docs
            ]
        }
        
        # Add LLM analysis if available
        # ... (continues with original logic)
        
        return analysis
    
    agent.analyze_topic = improved_analyze
    return agent


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    from multi_agent_system_simple import RAGKnowledgeBase
    
    # Initialize base knowledge base
    kb = RAGKnowledgeBase()
    
    # Create improved version
    improved_kb = ImprovedRAGKnowledgeBase(kb)
    
    # Test query expansion
    expander = QueryExpander()
    query = "startup funding"
    expanded = expander.expand_with_keywords(query, "investor")
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")
    
    # Test re-ranking
    test_docs = [
        {'content': 'Startup funding stages include seed, Series A, B', 'relevance': 0.85},
        {'content': 'General business information', 'relevance': 0.60},
        {'content': 'Funding metrics and valuation data', 'relevance': 0.75}
    ]
    
    reranker = SimpleReRanker()
    reranked = reranker.rerank("startup funding stages", test_docs, top_k=2)
    print(f"\nRe-ranked results:")
    for doc in reranked:
        print(f"  {doc['content'][:50]}... (relevance: {doc.get('relevance', 0)})")


