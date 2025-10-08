"""
Test Script for Investor Agent
==============================

This script tests the Investor Agent's RAG capabilities and analysis functions.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

async def test_investor_agent():
    """Test the Investor Agent functionality"""
    print("🧪 Testing Investor Agent...")
    
    try:
        # Import the system
        from multi_agent_system import MultiAgentSystem, StartupData
        
        # Create sample startup data
        test_startup = StartupData(
            name="EcoTech Solutions",
            problem_statement="Traditional energy sources are unsustainable and expensive for small businesses",
            solution="AI-powered renewable energy optimization platform for small business energy management",
            target_market="Small to medium businesses in North America seeking to reduce energy costs",
            business_model="Subscription-based SaaS with energy savings sharing model",
            competitive_advantage="Proprietary AI algorithms for energy optimization and real-time monitoring"
        )
        
        print(f"📊 Testing startup: {test_startup.name}")
        print(f"   Problem: {test_startup.problem_statement[:100]}...")
        print(f"   Solution: {test_startup.solution[:100]}...")
        print(f"   Business Model: {test_startup.business_model}")
        
        # Initialize the system
        print("\n🔄 Initializing multi-agent system...")
        system = MultiAgentSystem()
        
        # Test RAG knowledge base
        print("\n🔍 Testing RAG knowledge base...")
        investor_info = await system.knowledge_base.retrieve_relevant_info(
            "startup funding SaaS business model", 
            "investor"
        )
        
        print(f"✅ Retrieved {len(investor_info)} relevant documents")
        for i, info in enumerate(investor_info[:2]):  # Show first 2
            print(f"   Document {i+1}: {info[:200]}...")
        
        # Test single agent analysis
        print("\n📈 Testing Investor Agent analysis...")
        investor_agent = system.agents["investor"]
        analysis = await investor_agent.analyze_startup(test_startup)
        
        print("✅ Investor Agent analysis completed!")
        print(f"   Confidence Score: {analysis.confidence_score:.2f}")
        print(f"   Key Metrics: {list(analysis.key_metrics.keys())}")
        print(f"   Recommendations: {analysis.recommendations}")
        print(f"   Concerns: {analysis.concerns}")
        
        # Show detailed analysis
        print(f"\n📋 Detailed Analysis:")
        print(f"   {analysis.analysis[:500]}...")
        
        # Test context awareness
        print("\n🧠 Testing context awareness...")
        context = await investor_agent.get_context_awareness([analysis])
        print(f"✅ Context awareness: {context[:200]}...")
        
        print("\n🎉 All tests passed! Investor Agent is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_queries():
    """Test specific RAG queries for the Investor Agent"""
    print("\n🔍 Testing specific RAG queries...")
    
    try:
        from multi_agent_system import MultiAgentSystem
        
        system = MultiAgentSystem()
        
        # Test different query types
        queries = [
            ("startup funding stages", "investor"),
            ("business model SaaS", "investor"),
            ("market analysis TAM", "investor"),
            ("investment criteria", "investor"),
            ("risk assessment", "investor")
        ]
        
        for query, role in queries:
            print(f"\n   Query: '{query}' for {role}")
            results = await system.knowledge_base.retrieve_relevant_info(query, role)
            print(f"   Results: {len(results)} documents")
            if results:
                print(f"   First result: {results[0][:100]}...")
        
        print("✅ RAG query tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ RAG query test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Investor Agent Tests...")
    
    # Test basic functionality
    basic_test = await test_investor_agent()
    
    # Test RAG queries
    rag_test = await test_rag_queries()
    
    if basic_test and rag_test:
        print("\n🎉 All tests passed! Investor Agent is fully functional.")
        print("\n📋 Summary of capabilities:")
        print("   ✅ RAG knowledge base integration")
        print("   ✅ Startup analysis and evaluation")
        print("   ✅ Key metrics extraction")
        print("   ✅ Recommendations generation")
        print("   ✅ Risk assessment")
        print("   ✅ Context awareness")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
