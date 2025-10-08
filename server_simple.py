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

from multi_agent_system_simple import RAGMultiAgentSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.on_event("startup")
async def startup_event():
    """Initialize the multi-agent system on startup"""
    global multi_agent_system
    try:
        logger.info("Initializing RAG-powered multi-agent system...")
        multi_agent_system = RAGMultiAgentSystem()
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

        # Run analysis
        result = multi_agent_system.run_analysis(request.topic)

        logger.info(f"Analysis completed for topic: {request.topic}")

        return AnalysisResponse(
            status="success", 
            message="Topic analysis completed successfully", 
            data=result
        )

    except Exception as e:
        logger.error(f"Error during topic analysis: {e}")
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

@app.get("/healthz")
async def healthz():
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run("server_simple:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
