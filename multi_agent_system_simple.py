#!/usr/bin/env python3
"""
Simplified Multi-Agent System with PDF Support
==============================================

A streamlined version that focuses on core functionality without complex dependencies.
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
import requests

# Local LLM imports
try:
    from ctransformers import AutoModelForCausalLM
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

# RAG and Vector Database imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log LLM availability
if not LOCAL_LLM_AVAILABLE:
    logger.warning("ctransformers not available - local LLM disabled")

class PDFProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from a PDF file"""
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                return text.strip() if text else None
                
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return None
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Get PDF metadata"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    'pages': len(reader.pages),
                    'file_size': os.path.getsize(pdf_path),
                    'file_name': os.path.basename(pdf_path)
                }
                
                if reader.metadata:
                    for key, value in reader.metadata.items():
                        if value:
                            metadata[key] = str(value)
                
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to get metadata from {pdf_path}: {e}")
            return {'error': str(e)}

class LocalLLM:
    """Local LLM using ctransformers for Mistral 7B"""
    
    def __init__(self, model_path: str = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        self.model_path = model_path
        self.llm = None
        self.loaded = False
        
    def load_model(self) -> bool:
        """Load the local LLM model"""
        if not LOCAL_LLM_AVAILABLE:
            logger.error("ctransformers not available - cannot load local LLM")
            return False
            
        try:
            logger.info(f"Loading local LLM model: {self.model_path}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type="mistral",
                gpu_layers=0,  # CPU only
                threads=4,
                context_length=4096,
                batch_size=1
            )
            self.loaded = True
            logger.info("Local LLM model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from local LLM"""
        if not self.loaded or not self.llm:
            return "Error: Local LLM not loaded"
        
        try:
            # Format prompt for Mistral
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Generate response
            response = self.llm(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "\n\n", "[/INST]"],
                stream=False
            )
            
            # Clean up response
            if isinstance(response, list):
                response = response[0] if response else ""
            
            # Remove input prompt from response
            if formatted_prompt in response:
                response = response[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating local LLM response: {e}")
            return f"Error: {e}"

class RAGKnowledgeBase:
    """RAG-powered knowledge base with vector search and semantic retrieval"""
    
    def __init__(self, knowledge_dir: str = "knowledge_base", ollama_url: str = "http://localhost:11434"):
        self.knowledge_dir = Path(knowledge_dir)
        self.pdf_processor = PDFProcessor()
        self.ollama_url = ollama_url
        self.documents = {}
        
        # Initialize RAG components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="startup_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with RAG capabilities"""
        logger.info("Initializing RAG knowledge base...")
        
        # Check if collection already has data
        if self.collection.count() > 0:
            logger.info(f"Found existing vector database with {self.collection.count()} chunks")
            return
        
        # Load and process documents
        all_documents = []
        
        # Load text files
        text_files = list(self.knowledge_dir.glob("**/*.txt"))
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': str(text_file),
                            'type': 'text',
                            'file_name': text_file.name
                        }
                    )
                    all_documents.append(doc)
                    self.documents[str(text_file)] = {
                        'type': 'text',
                        'content': content,
                        'source': str(text_file),
                        'size': len(content)
                    }
            except Exception as e:
                logger.error(f"Failed to load text file {text_file}: {e}")
        
        # Load PDF files
        pdf_files = list(self.knowledge_dir.glob("**/*.pdf"))
        for pdf_file in pdf_files:
            try:
                text_content = self.pdf_processor.extract_text_from_pdf(str(pdf_file))
                metadata = self.pdf_processor.get_pdf_metadata(str(pdf_file))
                
                if text_content:
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            'source': str(pdf_file),
                            'type': 'pdf',
                            'file_name': pdf_file.name,
                            'pages': metadata.get('pages', 0)
                        }
                    )
                    all_documents.append(doc)
                    self.documents[str(pdf_file)] = {
                        'type': 'pdf',
                        'content': text_content,
                        'source': str(pdf_file),
                        'metadata': metadata,
                        'size': len(text_content)
                    }
                    logger.info(f"Loaded PDF: {pdf_file.name} ({metadata.get('pages', '?')} pages)")
                else:
                    logger.warning(f"No text extracted from PDF: {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"Failed to load PDF file {pdf_file}: {e}")
        
        # Split documents into chunks and create embeddings
        logger.info("Creating document chunks and embeddings...")
        all_chunks = []
        for doc in all_documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        # Add chunks to vector database
        if all_chunks:
            texts = [chunk.page_content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]
            ids = [f"chunk_{i}" for i in range(len(all_chunks))]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(all_chunks)} chunks to vector database")
        
        logger.info(f"RAG knowledge base initialized with {len(self.documents)} documents and {len(all_chunks)} chunks")
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """RAG-based semantic search through documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1 - distance
                
                formatted_results.append({
                    'source': metadata['source'],
                    'type': metadata['type'],
                    'relevance': similarity,
                    'preview': doc[:200] + "..." if len(doc) > 200 else doc,
                    'metadata': metadata,
                    'content': doc
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            # Fallback to simple search
            return self._fallback_search(query, max_results)

    def search_documents_for_role(self, query: str, role: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Role-aware retrieval: expand query and rerank results based on role-specific signals."""
        # Expand the query with role intent and repeated keywords to bias embeddings strongly
        role_intents = {
            "Investor": "Focus on ROI, revenue, margins, market size (TAM/SAM), funding, valuation, risk, competition.",
            "Researcher": "Focus on technical feasibility, state of the art, methods, benchmarks, datasets, limitations, innovation.",
            "User": "Focus on user experience, adoption, onboarding, retention, feedback, usability, pricing, support."
        }
        role_repeats = {
            "Investor": " ROI ROI revenue revenue market size TAM SAM valuation risk competition growth unit economics ",
            "Researcher": " technical feasibility benchmark dataset accuracy architecture methods innovation limitations ",
            "User": " user experience UX adoption onboarding retention feedback usability pricing support satisfaction "
        }
        expanded_query = f"{query}. {role_intents.get(role, '')} {role_repeats.get(role, '')}"

        # Get a larger candidate set, then rerank
        candidate_pool = max(10, max_results * 3)
        candidates = self.search_documents(expanded_query, max_results=candidate_pool)

        # Role keyword boosts (checked against full chunk content, not just preview)
        role_keywords = {
            "Investor": [
                "roi","revenue","profit","margin","tam","sam","market size","valuation","funding",
                "cost","risk","competition","cagr","growth","unit economics","payback"
            ],
            "Researcher": [
                "state of the art","algorithm","accuracy","benchmark","dataset","architecture",
                "innovation","technical","feasibility","limitations","research","study","evidence"
            ],
            "User": [
                "user","ux","adoption","retention","onboarding","feedback","ease","experience",
                "design","accessibility","pricing","support","satisfaction"
            ]
        }

        filename_hints = {
            "Investor": ["market","due diligence","finance","investment","sizing","sam","tam","pricing"],
            "Researcher": ["technology","trend","architecture","methods","benchmark","research"],
            "User": ["checklist","guide","user","design","ux","adoption"]
        }

        keywords = role_keywords.get(role, [])
        hints = filename_hints.get(role, [])

        def boost_score(item: Dict[str, Any]) -> float:
            text = (item.get('content') or item.get('preview') or '')
            source = (item.get('source') or '').lower()
            base = float(item.get('relevance', 0.0))

            # Keyword match boost (cap total keyword boost)
            kw_matches = 0
            low_text = text.lower()
            for kw in keywords:
                # count occurrences rather than presence
                kw_matches += low_text.count(kw)
            kw_boost = min(0.50, kw_matches * 0.01)  # up to +0.50 based on frequency

            # Filename/source hint boost
            hint_boost = 0.0
            for h in hints:
                if h in source:
                    hint_boost += 0.05
            hint_boost = min(hint_boost, 0.20)  # cap +0.20

            score = base + kw_boost + hint_boost
            # allow scores above 1 before normalization; we'll rescale later
            return max(0.0, score)

        # Compute boosted scores
        for c in candidates:
            c['role_relevance'] = boost_score(c)

        # Simple diversification: prefer unique sources, penalize duplicates
        selected: List[Dict[str, Any]] = []
        seen_sources = set()
        # Sort by role_relevance first
        candidates.sort(key=lambda x: x.get('role_relevance', x.get('relevance', 0.0)), reverse=True)
        for c in candidates:
            src = c.get('source') or ''
            # apply penalty if same source already selected
            penalty = 0.0 if src not in seen_sources else 0.25
            c['_final_score'] = c['role_relevance'] - penalty
        # Final sort after penalty
        candidates.sort(key=lambda x: x.get('_final_score', 0.0), reverse=True)

        for c in candidates:
            src = c.get('source') or ''
            if len(selected) >= max_results:
                break
            # enforce at least 2 unique sources when possible
            if src in seen_sources and len(seen_sources) < 2:
                continue
            selected.append(c)
            seen_sources.add(src)

        # If not enough, fill remaining regardless of source
        if len(selected) < max_results:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= max_results:
                        break

        # Normalize role_relevance to 0..1 for UI consistency
        max_score = max((x.get('role_relevance', 0.0) for x in selected), default=1.0)
        if max_score > 0:
            for x in selected:
                x['relevance'] = min(1.0, x.get('role_relevance', 0.0) / max_score)

        return selected[:max_results]
    
    def _fallback_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Fallback simple text search if RAG fails"""
        query_lower = query.lower()
        results = []
        
        for doc_id, doc_info in self.documents.items():
            content = doc_info['content'].lower()
            
            if query_lower in content:
                relevance = content.count(query_lower)
                results.append({
                    'source': doc_info['source'],
                    'type': doc_info['type'],
                    'relevance': relevance,
                    'preview': doc_info['content'][:200] + "..." if len(doc_info['content']) > 200 else doc_info['content'],
                    'metadata': doc_info.get('metadata', {})
                })
        
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:max_results]
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base contents"""
        summary = {
            'total_documents': len(self.documents),
            'by_type': {},
            'total_size': 0,
            'sources': []
        }
        
        for doc_info in self.documents.values():
            doc_type = doc_info['type']
            if doc_type not in summary['by_type']:
                summary['by_type'][doc_type] = 0
            summary['by_type'][doc_type] += 1
            
            summary['total_size'] += doc_info['size']
            summary['sources'].append(doc_info['source'])
        
        return summary

class RAGAgent:
    """RAG-powered agent with LLM integration"""
    
    def __init__(self, name: str, role: str, knowledge_base: RAGKnowledgeBase, local_llm: Optional[LocalLLM] = None, ollama_url: str = "http://localhost:11434"):
        self.name = name
        self.role = role
        self.knowledge_base = knowledge_base
        self.local_llm = local_llm
        self.ollama_url = ollama_url
        self.analysis_history = []
        # Optional quantitative model for investor
        self.success_model = None
        if self.role == "Investor":
            try:
                from models.startup_success_model import StartupSuccessModel
                self.success_model = StartupSuccessModel()
                if not self.success_model.load():
                    # Model missing; leave as None, handled gracefully
                    self.success_model = None
            except Exception as _:
                self.success_model = None
    
    def analyze_topic(self, topic: str) -> Dict[str, Any]:
        """Analyze a topic using RAG + LLM"""
        logger.info(f"Agent {self.name} analyzing topic: {topic}")
        
        # Search knowledge base using role-aware RAG to tailor documents per agent
        try:
            relevant_docs = self.knowledge_base.search_documents_for_role(topic, self.role, max_results=5)
        except Exception:
            relevant_docs = self.knowledge_base.search_documents(topic, max_results=5)
        
        analysis = {
            'agent': self.name,
            'role': self.role,
            'topic': topic,
            'timestamp': time.time(),
            'relevant_documents': len(relevant_docs),
            'insights': [],
            'recommendations': [],
            'llm_analysis': None
        }
        
        if relevant_docs:
            # Create role-specific prompt
            prompt = self._create_role_prompt(topic, relevant_docs)
            
            # Get LLM analysis
            llm_response = self._call_llm(prompt)
            analysis['llm_analysis'] = llm_response or "LLM analysis unavailable"
            
            # Add document context
            analysis['document_previews'] = [
                {
                    'source': doc['source'],
                    'type': doc['type'],
                    'preview': doc['preview'],
                    'relevance': doc['relevance']
                }
                for doc in relevant_docs
            ]

            # Quantitative augmentation for Investor
            if self.role == "Investor" and self.success_model is not None:
                try:
                    from models.startup_success_model import StartupFeatures
                    features = self.success_model.extract_features_from_text(topic)
                    success_prob = self.success_model.predict_proba(features)
                    analysis['quantitative_model'] = {
                        'type': 'RandomForestClassifier',
                        'success_probability': round(success_prob * 100, 1),
                        'features': {
                            'sector': features.sector,
                            'team_size': features.team_size,
                            'funding_stage': features.funding_stage,
                            'region': features.region,
                            'market_competitiveness': features.market_competitiveness
                        }
                    }
                    analysis['insights'].append(f"Model-estimated success probability: {analysis['quantitative_model']['success_probability']}%")
                except Exception as e:
                    logger.warning(f"Investor quantitative model unavailable: {e}")
        else:
            analysis['insights'].append("No relevant documents found for this topic")
            analysis['recommendations'].append("Consider expanding knowledge base with relevant materials")
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _create_role_prompt(self, topic: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Create role-specific prompt for LLM"""
        # Combine relevant document content
        doc_context = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['content'][:500]}..."
            for doc in relevant_docs
        ])
        
        role_prompts = {
            "Investor": f"""
            As an INVESTOR, analyze this startup topic: "{topic}"
            
            Focus on:
            - Market opportunity and size
            - Competitive landscape and differentiation
            - Financial viability and ROI potential
            - Risk assessment and mitigation
            - Investment recommendation (Invest/Pass/More Info)
            
            Use the following context from your knowledge base:
            {doc_context}
            
            Provide a structured analysis with specific insights and actionable recommendations.
            """,
            "Researcher": f"""
            As a RESEARCHER, analyze this startup topic: "{topic}"
            
            Focus on:
            - Market trends and growth potential
            - Target customer segments and needs
            - Competitive analysis and positioning
            - Technology and innovation opportunities
            - Research gaps and future directions
            
            Use the following context from your knowledge base:
            {doc_context}
            
            Provide data-driven insights and research recommendations.
            """,
            "User": f"""
            As a USER/ENTREPRENEUR, analyze this startup topic: "{topic}"
            
            Focus on:
            - Practical implementation challenges
            - User experience and market fit
            - Resource requirements and constraints
            - Go-to-market strategy
            - Success factors and common pitfalls
            
            Use the following context from your knowledge base:
            {doc_context}
            
            Provide practical advice and actionable next steps.
            """
        }
        
        return role_prompts.get(self.role, f"Analyze this topic: {topic}\n\nContext: {doc_context}")
    
    def _call_llm(self, prompt: str, model: str = "mistral-local") -> Optional[str]:
        """Call LLM (local or Ollama)"""
        # Try local LLM first; lazy-init if necessary
        if (self.local_llm is None or not getattr(self.local_llm, 'loaded', False)) and LOCAL_LLM_AVAILABLE:
            try:
                logger.info("Lazy-loading local LLM model for agent")
                self.local_llm = LocalLLM()
                self.local_llm.load_model()
            except Exception as _:
                self.local_llm = None
        if self.local_llm and self.local_llm.loaded:
            logger.info("Using local LLM for analysis")
            return self.local_llm.generate_response(prompt, max_tokens=256)
        
        # Fallback to Ollama
        try:
            logger.info("Using Ollama API for analysis")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            return None
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return self.analysis_history

class RAGMultiAgentSystem:
    """RAG-powered multi-agent system with LLM integration"""
    
    def __init__(self, knowledge_dir: str = "knowledge_base", ollama_url: str = "http://localhost:11434", use_local_llm: bool = True):
        self.knowledge_base = RAGKnowledgeBase(knowledge_dir, ollama_url)
        self.ollama_url = ollama_url
        self.agents = {}
        
        # Initialize local LLM if requested and available
        self.local_llm = None
        if use_local_llm and LOCAL_LLM_AVAILABLE:
            self.local_llm = LocalLLM()
            if not self.local_llm.load_model():
                logger.warning("Failed to load local LLM, falling back to Ollama")
                self.local_llm = None
        else:
            logger.info("Using Ollama for LLM responses")
        
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize the agent team with correct roles"""
        self.agents = {
            'investor': RAGAgent("Investor Agent", "Investor", self.knowledge_base, self.local_llm, self.ollama_url),
            'researcher': RAGAgent("Researcher Agent", "Researcher", self.knowledge_base, self.local_llm, self.ollama_url),
            'user': RAGAgent("User Agent", "User", self.knowledge_base, self.local_llm, self.ollama_url)
        }
        logger.info(f"Initialized {len(self.agents)} RAG-powered agents")
    
    def run_analysis(self, topic: str) -> Dict[str, Any]:
        """Run analysis with all agents"""
        logger.info(f"Running multi-agent analysis on: {topic}")
        
        results = {
            'topic': topic,
            'timestamp': time.time(),
            'agent_analyses': {},
            'summary': {}
        }
        
        # Run analysis with each agent
        for agent_id, agent in self.agents.items():
            analysis = agent.analyze_topic(topic)
            results['agent_analyses'][agent_id] = analysis
        
        # Generate summary
        total_docs = sum(analysis['relevant_documents'] for analysis in results['agent_analyses'].values())
        results['summary'] = {
            'total_agents': len(self.agents),
            'total_relevant_documents': total_docs,
            'analysis_complete': True
        }
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and knowledge base summary"""
        kb_summary = self.knowledge_base.get_document_summary()
        
        return {
            'status': 'operational',
            'agents': len(self.agents),
            'knowledge_base': kb_summary,
            'timestamp': time.time()
        }

def main():
    """Main function to demonstrate the RAG + LLM system"""
    print("🚀 RAG-Powered Multi-Agent System with LLM Integration")
    print("=" * 70)
    
    # Initialize system
    print("🔄 Initializing RAG-powered multi-agent system...")
    system = RAGMultiAgentSystem()
    
    # Check LLM status
    if system.local_llm and system.local_llm.loaded:
        print("✅ Local LLM (Mistral 7B) loaded and ready")
    else:
        print("⚠️ Local LLM not available, checking Ollama...")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    print(f"✅ Ollama available with {len(models)} models")
                else:
                    print("⚠️ Ollama running but no models installed")
            else:
                print("❌ Ollama not responding properly")
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            print("💡 Using RAG-only mode (no LLM responses)")
    
    # Show system status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"  Agents: {status['agents']}")
    print(f"  Total Documents: {status['knowledge_base']['total_documents']}")
    print(f"  Document Types: {status['knowledge_base']['by_type']}")
    print(f"  RAG Enabled: True")
    print(f"  LLM Enabled: True")
    
    # Run sample analysis
    print(f"\n🔍 Running RAG + LLM analysis...")
    results = system.run_analysis("AI startup funding")
    
    print(f"\n📈 Analysis Results:")
    for agent_id, analysis in results['agent_analyses'].items():
        print(f"\n  🤖 {analysis['agent']} ({analysis['role']}):")
        print(f"    Relevant documents: {analysis['relevant_documents']}")
        if analysis.get('llm_analysis'):
            print(f"    🧠 LLM Analysis: {analysis['llm_analysis'][:200]}...")
        for insight in analysis['insights']:
            print(f"    💡 {insight}")
        for rec in analysis['recommendations']:
            print(f"    💭 {rec}")
    
    print(f"\n✅ RAG + LLM system is ready for use!")
    print(f"💡 You can now analyze topics using semantic search and AI analysis")

if __name__ == "__main__":
    main()

