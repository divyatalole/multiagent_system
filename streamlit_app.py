#!/usr/bin/env python3
"""
StartupAI Streamlit Web Interface
==================================

User-facing web application for startup evaluation using the multi-agent system.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="StartupAI - AI-Powered Startup Evaluation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .investor-card { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .researcher-card { background-color: #f1f8e9; border-left: 5px solid #8bc34a; }
    .user-card { background-color: #fce4ec; border-left: 5px solid #e91e63; }
    .metric-box {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem;
    }
    .success-message { color: #4caf50; font-weight: bold; }
    .error-message { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@st.cache_data
def get_agents() -> List[Dict[str, str]]:
    """Get list of available agents"""
    try:
        response = requests.get(f"{API_BASE_URL}/agents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


def run_analysis(topic: str, max_results: int = 5) -> Optional[Dict[str, Any]]:
    """Run analysis using the multi-agent system"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"topic": topic, "max_results": max_results},
            timeout=600  # 10 minutes timeout - enough for quality LLM inference with meaningful output
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return result.get("data")
            else:
                st.error(f"Analysis failed: {result.get('message', 'Unknown error')}")
                return None
        else:
            st.error(f"API returned status {response.status_code}: {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Analysis timed out after 10 minutes. The system is generating detailed analysis - please check server logs or try again.")
        return None
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        return None


def run_followup(question: str, previous_state: Dict[str, Any], target_agent: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Handle follow-up question"""
    try:
        payload = {
            "question": question,
            "previous_state": previous_state,
        }
        if target_agent:
            payload["target_agent"] = target_agent
        
        response = requests.post(
            f"{API_BASE_URL}/analyze/followup",
            json=payload,
            timeout=600  # 10 minutes for followup questions with quality output
        )
        if response.status_code == 200:
            return response.json().get("data")
        return None
    except Exception as e:
        st.error(f"Error processing followup: {e}")
        return None


def display_agent_analysis(agent_id: str, analysis: Dict[str, Any]):
    """Display analysis from a specific agent"""
    agent_colors = {
        "investor": "investor-card",
        "researcher": "researcher-card",
        "user": "user-card"
    }
    
    agent_names = {
        "investor": "Investor Agent 💰",
        "researcher": "Researcher Agent 🔬",
        "user": "User Agent 👥"
    }
    
    css_class = agent_colors.get(agent_id, "agent-card")
    name = agent_names.get(agent_id, agent_id.title())
    
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    st.markdown(f"### {name}")
    
    # Show retrieved documents count
    num_docs = analysis.get("relevant_documents", 0)
    st.caption(f"📄 Retrieved {num_docs} relevant document(s)")
    
    # Display retrieved documents FIRST (most important)
    if "document_previews" in analysis and analysis["document_previews"]:
        st.markdown("**📚 Retrieved Documents:**")
        with st.expander(f"View {len(analysis['document_previews'])} documents", expanded=True):
            for i, doc in enumerate(analysis["document_previews"], 1):
                st.markdown(f"**Document {i}: {doc.get('source', 'Unknown Source')}**")
                preview = doc.get('preview', doc.get('content', ''))
                if preview:
                    st.text_area(
                        f"Preview {i}",
                        value=preview[:500] + ("..." if len(preview) > 500 else ""),
                        height=100,
                        key=f"{agent_id}_doc_{i}",
                        disabled=True,
                        label_visibility="collapsed"
                    )
                if 'relevance' in doc:
                    st.caption(f"Relevance Score: {doc['relevance']:.3f}")
                st.divider()
    else:
        st.warning("⚠️ No documents retrieved for this agent")
    
    # Display LLM Analysis (always show if present)
    if "llm_analysis" in analysis and analysis["llm_analysis"]:
        st.markdown("**🤖 LLM Analysis:**")
        llm_text = analysis["llm_analysis"]
        st.text_area(
            "Full Analysis",
            value=llm_text,
            height=200,
            key=f"{agent_id}_llm",
            disabled=True,
            label_visibility="collapsed"
        )
    else:
        st.info("💭 LLM analysis not available for this agent")
    
    # Display insights
    if "insights" in analysis and analysis["insights"]:
        st.markdown("**💡 Key Insights:**")
        for insight in analysis["insights"][:5]:
            st.markdown(f"• {insight}")
    elif num_docs == 0:
        st.info("💡 No insights available - no relevant documents found")
    
    # Display recommendations
    if "recommendations" in analysis and analysis["recommendations"]:
        st.markdown("**💼 Recommendations:**")
        for rec in analysis["recommendations"][:3]:
            st.markdown(f"→ {rec}")
    
    # Show quantitative models if available
    if "quantitative_model" in analysis:
        st.markdown("**📊 Quantitative Assessment:**")
        qm = analysis["quantitative_model"]
        st.metric("Success Probability", f"{qm.get('success_probability', 0)}%")
    
    if "market_trend_model" in analysis:
        st.markdown("**📈 Market Trend Forecast:**")
        mtm = analysis["market_trend_model"]
        st.metric("Next Period Forecast", f"{mtm.get('next_period', 'N/A')}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🚀 StartupAI</h1><p>AI-Powered Multi-Agent Startup Evaluation</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API health check
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info("Make sure the API server is running at http://localhost:8000")
            st.stop()
        
        st.divider()
        
        # System info
        st.subheader("System Info")
        try:
            status_response = requests.get(f"{API_BASE_URL}/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                st.metric("Knowledge Base", status.get("knowledge_base", {}).get("total_documents", 0))
        except Exception:
            pass
        
        st.divider()
        
        # About
        st.subheader("About")
        st.info("""
        **StartupAI** analyzes your startup idea using three specialized AI agents:
        
        - **Investor Agent**: Financial viability
        - **Researcher Agent**: Market research
        - **User Agent**: UX perspective
        
        Get comprehensive insights in seconds!
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 New Analysis", "💬 Conversation", "📊 Session History"])
    
    with tab1:
        st.header("Startup Evaluation Form")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            startup_description = st.text_area(
                "Describe your startup idea",
                height=200,
                placeholder="Example: AI-powered personal finance app that helps millennials save money automatically based on spending patterns and goals..."
            )
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Be specific about your solution
            - Mention target market
            - Describe key features
            - Include competitive advantages
            """)
        
        max_results = st.slider("Number of documents to retrieve", 3, 10, 5)
        
        if st.button("🚀 Analyze Startup", type="primary", use_container_width=True):
            if not startup_description:
                st.error("Please enter a startup description")
            else:
                with st.spinner("🔄 Analyzing with AI agents..."):
                    # Run analysis
                    result = run_analysis(startup_description, max_results)
                    
                    if result:
                        # Store in session state
                        if "conversation_history" not in st.session_state:
                            st.session_state.conversation_history = []
                        
                        conversation_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "topic": startup_description,
                            "results": result
                        }
                        st.session_state.conversation_history.append(conversation_entry)
                        st.session_state["latest_analysis"] = result
                        
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        st.divider()
                        st.header("📊 Analysis Results")
                        
                        # Show synthesis if available
                        if "_synthesis" in result.get("agent_analyses", {}):
                            synth = result["agent_analyses"]["_synthesis"]
                            st.subheader("Overall Assessment")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Overall Score", f"{synth.get('overall_score', 0):.1f}/10")
                            with col2:
                                st.metric("Risk Level", synth.get('risk_assessment', 'N/A'))
                            with col3:
                                st.metric("Feasibility", synth.get('technical_feasibility', 'N/A'))
                            with col4:
                                st.metric("Recommendation", synth.get('recommendation', 'N/A'))
                            st.divider()
                        
                        # Display agent analyses
                        agent_analyses = result.get("agent_analyses", {})
                        
                        # Debug: Show what agents we have
                        if not agent_analyses:
                            st.warning("⚠️ No agent analyses found in response. Response keys: " + ", ".join(result.keys()))
                            st.json(result)  # Show full response for debugging
                        else:
                            st.info(f"📊 Found analyses from {len(agent_analyses)} agent(s): {', '.join(agent_analyses.keys())}")
                        
                        # Display each agent (flexible to handle any agent ID format)
                        for agent_id, analysis in agent_analyses.items():
                            # Skip synthesis if it's a special key
                            if agent_id == "_synthesis":
                                continue
                            
                            # Normalize agent ID for display
                            normalized_id = agent_id.lower().replace(" agent", "").replace(" ", "_")
                            if normalized_id not in ["investor", "researcher", "user"]:
                                normalized_id = agent_id  # Use original if not recognized
                            
                            display_agent_analysis(normalized_id, analysis)
                    else:
                        st.error("Failed to complete analysis. Please try again.")
    
    with tab2:
        st.header("💬 Ask Follow-up Questions")
        
        if "latest_analysis" not in st.session_state:
            st.info("👆 Run an analysis first to start a conversation")
        else:
            st.success("✅ You have an active analysis session")
            
            # Show original topic
            st.markdown("**Original Topic:**")
            st.info(st.session_state["latest_analysis"].get("topic", "N/A"))
            
            st.divider()
            
            # Follow-up question form
            col1, col2 = st.columns([3, 1])
            
            with col1:
                followup_question = st.text_input(
                    "Ask a follow-up question",
                    placeholder="Example: What are the main risks? What about the market size?"
                )
            
            with col2:
                target_agent = st.selectbox(
                    "Target Agent",
                    ["All Agents", "Investor", "Researcher", "User"],
                    help="Choose which agent should answer"
                )
            
            if st.button("❓ Ask Question", type="primary"):
                if not followup_question:
                    st.error("Please enter a question")
                else:
                    # Map display name to agent ID
                    agent_map = {
                        "All Agents": None,
                        "Investor": "investor",
                        "Researcher": "researcher",
                        "User": "user"
                    }
                    
                    with st.spinner("🤔 Thinking..."):
                        # Prepare previous state
                        prev_state = {
                            "topic": st.session_state["latest_analysis"].get("topic"),
                            "agent_results": st.session_state["latest_analysis"].get("agent_analyses", {})
                        }
                        
                        result = run_followup(
                            followup_question,
                            prev_state,
                            agent_map[target_agent]
                        )
                        
                        if result:
                            st.success("✅ Response received!")
                            st.divider()
                            
                            # Display follow-up results
                            followup_results = result.get("followup_results", {})
                            
                            for agent_id, analysis in followup_results.items():
                                display_agent_analysis(agent_id, analysis)
                        else:
                            st.error("Failed to process question. Please try again.")
    
    with tab3:
        st.header("📊 Session History")
        
        if "conversation_history" not in st.session_state or not st.session_state.conversation_history:
            st.info("No analysis history yet. Run your first analysis to see it here.")
        else:
            st.success(f"📝 {len(st.session_state.conversation_history)} analyses completed")
            
            for idx, entry in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Analysis {len(st.session_state.conversation_history) - idx}: {entry['topic'][:100]}..."):
                    st.markdown(f"**Topic:** {entry['topic']}")
                    st.markdown(f"**Time:** {entry['timestamp']}")
                    st.markdown("**Results:**")
                    
                    if "agent_analyses" in entry["results"]:
                        agents = entry["results"]["agent_analyses"]
                        agent_count = len([k for k in agents.keys() if not k.startswith("_")])
                        st.success(f"✅ {agent_count} agents analyzed")
            
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.session_state["latest_analysis"] = None
                st.success("History cleared!")
                st.rerun()


if __name__ == "__main__":
    main()

