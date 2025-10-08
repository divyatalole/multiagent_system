#!/usr/bin/env python3
"""
Display Analysis Output in Readable Format
"""

import json
import sys
from pathlib import Path

def display_analysis():
    """Display the analysis output in a readable format"""
    
    # Try to read the saved output file
    output_file = Path("analysis_output.json")
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    else:
        print("❌ No analysis output file found. Run an analysis first.")
        return
    
    if data['status'] != 'success':
        print(f"❌ Analysis failed: {data.get('error', 'Unknown error')}")
        return
    
    analysis_data = data['data']
    topic = analysis_data['topic']
    
    print("=" * 80)
    print(f"🤖 MULTI-AGENT ANALYSIS: {topic.upper()}")
    print("=" * 80)
    
    # Display each agent's analysis
    for agent_id, analysis in analysis_data['agent_analyses'].items():
        print(f"\n🔹 {analysis['agent']} ({analysis['role']})")
        print("-" * 60)
        
        # Show relevant documents
        print(f"📚 Found {analysis['relevant_documents']} relevant documents:")
        for i, doc in enumerate(analysis['document_previews'][:3], 1):
            print(f"   {i}. {doc['source']} (relevance: {doc['relevance']:.3f})")
        
        # Show LLM analysis
        if analysis.get('llm_analysis'):
            print(f"\n🧠 AI Analysis:")
            print(f"   {analysis['llm_analysis']}")
        
        # Show insights and recommendations if any
        if analysis.get('insights'):
            print(f"\n💡 Insights:")
            for insight in analysis['insights']:
                print(f"   • {insight}")
        
        if analysis.get('recommendations'):
            print(f"\n🎯 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
        
        print()
    
    # Summary
    summary = analysis_data['summary']
    print("=" * 80)
    print(f"📊 SUMMARY: {summary['total_agents']} agents analyzed {summary['total_relevant_documents']} documents")
    print("=" * 80)

if __name__ == "__main__":
    display_analysis()
