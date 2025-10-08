#!/usr/bin/env python3
"""
Quick test of LLM integration without loading the full model
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_llm_availability():
    """Test if LLM components are available"""
    print("🔍 Testing LLM Integration...")
    
    # Test ctransformers
    try:
        from ctransformers import AutoModelForCausalLM
        print("✅ ctransformers available")
        ctransformers_available = True
    except ImportError as e:
        print(f"❌ ctransformers not available: {e}")
        ctransformers_available = False
    
    # Test model file
    model_path = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)  # GB
        print(f"✅ Model file found: {model_path} ({file_size:.1f} GB)")
        model_available = True
    else:
        print(f"❌ Model file not found: {model_path}")
        model_available = False
    
    # Test RAG system without LLM
    try:
        from multi_agent_system_simple import RAGMultiAgentSystem
        print("✅ RAG system import successful")
        
        # Initialize without LLM
        print("🔄 Initializing RAG system (without LLM)...")
        system = RAGMultiAgentSystem(use_local_llm=False)
        print("✅ RAG system initialized successfully")
        
        # Test RAG search
        print("🔍 Testing RAG search...")
        results = system.knowledge_base.search_documents("AI startup funding", max_results=3)
        print(f"✅ RAG search found {len(results)} relevant documents")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['source']} (relevance: {result['relevance']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False

def test_quick_llm():
    """Test LLM loading with minimal settings"""
    if not os.path.exists("model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        print("❌ Model file not found")
        return False
    
    try:
        from ctransformers import AutoModelForCausalLM
        print("🔄 Loading LLM model (this may take a moment)...")
        
        # Load with minimal settings for quick test
        llm = AutoModelForCausalLM.from_pretrained(
            "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            model_type="mistral",
            gpu_layers=0,
            threads=2,  # Reduced threads
            context_length=2048,  # Smaller context
            batch_size=1
        )
        
        print("✅ LLM model loaded successfully")
        
        # Quick test
        test_prompt = "What is artificial intelligence?"
        print(f"🧠 Testing prompt: {test_prompt}")
        
        response = llm(
            f"<s>[INST] {test_prompt} [/INST]",
            max_new_tokens=50,  # Short response
            temperature=0.7,
            stop=["</s>", "\n\n"]
        )
        
        print(f"✅ LLM Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 LLM Integration Test")
    print("=" * 50)
    
    # Test 1: Basic availability
    rag_ok = test_llm_availability()
    
    print("\n" + "=" * 50)
    
    # Test 2: Quick LLM test (optional)
    if input("\n🤔 Test LLM loading? (y/n): ").lower() == 'y':
        llm_ok = test_quick_llm()
    else:
        llm_ok = False
        print("⏭️ Skipping LLM test")
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  RAG System: {'✅ Working' if rag_ok else '❌ Failed'}")
    print(f"  LLM System: {'✅ Working' if llm_ok else '❌ Not tested/Failed'}")
    
    if rag_ok:
        print("\n🎉 Your RAG system is working! You can use it with or without LLM.")
    else:
        print("\n❌ There are issues with the RAG system that need to be fixed.")

