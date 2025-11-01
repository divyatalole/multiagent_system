"""
Simplified FastAPI Server for Multi-Agent System
================================================

A streamlined server that uses the simplified multi-agent system without complex dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import uvicorn

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from multi_agent_system_simple import RAGMultiAgentSystem

# Try to import orchestrator, but make it optional
try:
    from orchestrator import MultiAgentOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except Exception:
    ORCHESTRATOR_AVAILABLE = False
    logger.warning("Orchestrator not available (langgraph not installed). Followup functionality disabled.")

# Initialize FastAPI app
app = FastAPI(
    title="StartupAI Multi-Agent API (Simplified)",
    description="AI-powered startup evaluation using multiple specialized agents (simplified version)",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    topic: str
    max_results: int = 5

class FollowupRequest(BaseModel):
    question: str
    previous_state: Dict[str, Any]
    target_agent: Optional[str] = None

class AnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Global variables
multi_agent_system: Optional[RAGMultiAgentSystem] = None
orchestrator = None  # Will be set during startup

@app.on_event("startup")
async def startup_event():
    """Initialize the multi-agent system on startup"""
    global multi_agent_system, orchestrator
    try:
        logger.info("Initializing RAG-powered multi-agent system...")
        multi_agent_system = RAGMultiAgentSystem()
        if ORCHESTRATOR_AVAILABLE:
            orchestrator = MultiAgentOrchestrator(multi_agent_system)
            logger.info("Orchestrator initialized with followup support")
        else:
            orchestrator = None
            logger.info("Orchestrator not available - followup disabled")
        logger.info("RAG-powered multi-agent system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize multi-agent system: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat(), version="1.0.0")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global multi_agent_system

    if multi_agent_system is None:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")

    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat(), version="1.0.0")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_topic(request: AnalysisRequest):
    """Analyze a topic using all agents"""
    global multi_agent_system

    if multi_agent_system is None:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")

    try:
        logger.info(f"Starting analysis for topic: {request.topic}")

        # Run analysis in thread pool with timeout protection (non-blocking)
        import asyncio
        import concurrent.futures
        
        def run_analysis_sync():
            return multi_agent_system.run_analysis(request.topic)
        
        try:
            # Run in executor with 600 second timeout (10 minutes total for all 3 agents)
            # This allows enough time for quality LLM output (300s per agent max)
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, run_analysis_sync),
                    timeout=600.0  # 10 minutes total - enough for quality analysis
                )
        except asyncio.TimeoutError:
            logger.error(f"Analysis timed out after 600s for topic: {request.topic}")
            raise HTTPException(
                status_code=504,
                detail="Analysis timed out after 10 minutes. Please check server logs or try again."
            )

        logger.info(f"Analysis completed for topic: {request.topic}")

        return AnalysisResponse(
            status="success", 
            message="Topic analysis completed successfully", 
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during topic analysis: {e}", exc_info=True)
        return AnalysisResponse(
            status="error", 
            message="Failed to analyze topic", 
            error=str(e)
        )

@app.get("/agents", response_model=List[Dict[str, str]])
async def list_agents():
    """List all available AI agents"""
    global multi_agent_system

    if multi_agent_system is None:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")

    agents = []
    for name, agent in multi_agent_system.agents.items():
        agents.append({
            "name": name,
            "role": agent.role,
            "description": f"{agent.role.title()} agent specializing in {agent.role}-specific analysis",
        })

    return agents

@app.get("/status")
async def get_system_status():
    """Get system status and knowledge base summary"""
    global multi_agent_system

    if multi_agent_system is None:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")

    try:
        status = multi_agent_system.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/files")
async def list_kb_files():
    """List all files in the knowledge base directory"""
    from pathlib import Path
    kb_dir = Path("knowledge_base")
    files = []
    for path in kb_dir.rglob("*"):
        if path.is_file():
            files.append({
                "name": path.name, 
                "path": str(path.relative_to(kb_dir)), 
                "size": path.stat().st_size
            })
    return files

@app.post("/analyze/followup", response_model=AnalysisResponse)
async def followup_analysis(request: FollowupRequest):
    """Handle follow-up questions in an interactive dialogue"""
    global multi_agent_system, orchestrator

    if multi_agent_system is None:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")

    try:
        logger.info(f"Processing followup question: {request.question}")

        # If orchestrator available, use it
        if orchestrator is not None:
            result = orchestrator.follow_up(
                prev_state=request.previous_state,
                question=request.question,
                target=request.target_agent
            )
        else:
            # Fallback: Simple followup without orchestrator
            if request.target_agent:
                # Route to specific agent
                agent = multi_agent_system.agents.get(request.target_agent)
                if agent:
                    result = {
                        'topic': request.previous_state.get('topic', ''),
                        'followup_question': request.question,
                        'followup_results': {request.target_agent: agent.analyze_topic(request.question)}
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown agent: {request.target_agent}")
            else:
                # Analyze with all agents
                results = {}
                for agent_id, agent in multi_agent_system.agents.items():
                    results[agent_id] = agent.analyze_topic(request.question)
                result = {
                    'topic': request.previous_state.get('topic', ''),
                    'followup_question': request.question,
                    'followup_results': results
                }

        logger.info("Followup analysis completed")

        return AnalysisResponse(
            status="success",
            message="Followup analysis completed successfully",
            data=result
        )

    except Exception as e:
        logger.error(f"Error during followup analysis: {e}")
        return AnalysisResponse(
            status="error",
            message="Failed to process followup question",
            error=str(e)
        )

@app.get("/healthz")
async def healthz():
    return {"ok": True}

if __name__ == "__main__":
    import asyncio
    import logging
    
    # Fix for Windows asyncio connection reset errors
    if sys.platform == 'win32':
        # Suppress the connection reset error logging (it's harmless)
        logging.getLogger('asyncio').setLevel(logging.ERROR)
        
        # Set proper event loop policy for Windows
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except AttributeError:
            # Fallback for older Python versions
            pass
    
    # Configure uvicorn to handle Windows connections better
    uvicorn.run(
        "server_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )
