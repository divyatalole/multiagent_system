#!/usr/bin/env python3
"""
RAG Verification Test Script
============================
Tests the complete RAG + LLM + Quantitative Model pipeline
"""

import requests
import json
import time

def test_rag_system():
    """Test the complete RAG system"""
    print("🔍 RAG + LLM + Quantitative Model Verification")
    print("=" * 60)
    
    # Test API health
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API Server: Healthy")
        else:
            print("❌ API Server: Unhealthy")
            return
    except Exception as e:
        print(f"❌ API Server: Connection failed - {e}")
        return
    
    # Test knowledge base status
    try:
        status_response = requests.get("http://localhost:8000/status", timeout=10)
        if status_response.status_code == 200:
            status_data = status_response.json()
            kb_info = status_data.get('knowledge_base', {})
            print(f"✅ Knowledge Base: {kb_info.get('total_documents', 0)} documents loaded")
            print(f"   Document types: {kb_info.get('by_type', {})}")
        else:
            print("❌ Knowledge Base: Status check failed")
    except Exception as e:
        print(f"⚠️ Knowledge Base: Status check error - {e}")
    
    # Test RAG + LLM + Quantitative Model
    print("\n🧠 Testing RAG + LLM + Quantitative Model Pipeline...")
    test_topics = [
        "AI startup funding",
        "beauty tech innovation", 
        "fintech market analysis"
    ]
    
    for topic in test_topics:
        print(f"\n📊 Testing topic: '{topic}'")
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/analyze", 
                json={"topic": topic},
                timeout=120  # 2 minutes timeout for LLM processing
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                analysis_data = data.get('data', {})
                
                print(f"   ✅ Analysis completed in {end_time - start_time:.1f}s")
                print(f"   📈 Total relevant documents: {analysis_data.get('summary', {}).get('total_relevant_documents', 0)}")
                
                # Check each agent
                agent_analyses = analysis_data.get('agent_analyses', {})
                for agent_id, analysis in agent_analyses.items():
                    agent_name = analysis.get('agent', agent_id)
                    docs_found = analysis.get('relevant_documents', 0)
                    has_llm = bool(analysis.get('llm_analysis'))
                    has_quant = 'quantitative_model' in analysis
                    
                    print(f"   🤖 {agent_name}: {docs_found} docs, LLM: {'✅' if has_llm else '❌'}, Quant: {'✅' if has_quant else '❌'}")
                    
                    # Show quantitative model details for Investor
                    if has_quant and agent_id == 'investor':
                        quant_model = analysis['quantitative_model']
                        print(f"      📊 Success Probability: {quant_model['success_probability']}%")
                        print(f"      🎯 Model Type: {quant_model['type']}")
                        print(f"      🔧 Features: {quant_model['features']}")
                    
                    # Show RAG document previews
                    if analysis.get('document_previews'):
                        print(f"      📚 Top RAG documents:")
                        for i, doc in enumerate(analysis['document_previews'][:2]):
                            source_name = doc['source'].split('\\')[-1] if '\\' in doc['source'] else doc['source'].split('/')[-1]
                            print(f"         {i+1}. {source_name} (relevance: {doc['relevance']:.2f})")
                            print(f"            Preview: {doc['preview'][:80]}...")
                
            else:
                print(f"   ❌ Analysis failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Analysis timed out for '{topic}'")
        except Exception as e:
            print(f"   ❌ Analysis error for '{topic}': {e}")
    
    print("\n🎯 RAG System Verification Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_system()
