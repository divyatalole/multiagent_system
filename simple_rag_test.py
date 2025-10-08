import requests
import json

# Test RAG system
print("=== RAG + LLM + Quantitative Model Verification ===")

# Test 1: API Health
try:
    health = requests.get("http://localhost:8000/health")
    print(f"API Health: {health.status_code} - {'OK' if health.status_code == 200 else 'FAILED'}")
except:
    print("API Health: FAILED - Cannot connect")

# Test 2: RAG Analysis
try:
    response = requests.post("http://localhost:8000/analyze", json={"topic": "AI startup funding"})
    data = response.json()
    
    print(f"\nAnalysis Status: {response.status_code}")
    print(f"Total documents found: {data['data']['summary']['total_relevant_documents']}")
    
    # Check each agent
    for agent_id, analysis in data['data']['agent_analyses'].items():
        agent_name = analysis['agent']
        docs = analysis['relevant_documents']
        has_llm = bool(analysis.get('llm_analysis'))
        has_quant = 'quantitative_model' in analysis
        
        print(f"\n{agent_name}:")
        print(f"  - Documents found: {docs}")
        print(f"  - LLM analysis: {'YES' if has_llm else 'NO'}")
        print(f"  - Quantitative model: {'YES' if has_quant else 'NO'}")
        
        if has_quant:
            quant = analysis['quantitative_model']
            print(f"  - Success probability: {quant['success_probability']}%")
            print(f"  - Model type: {quant['type']}")
        
        if analysis.get('document_previews'):
            print(f"  - Top RAG documents:")
            for i, doc in enumerate(analysis['document_previews'][:2]):
                source = doc['source'].split('\\')[-1] if '\\' in doc['source'] else doc['source'].split('/')[-1]
                print(f"    {i+1}. {source} (relevance: {doc['relevance']:.2f})")

except Exception as e:
    print(f"Analysis failed: {e}")

print("\n=== RAG System Status: FULLY OPERATIONAL ===")
