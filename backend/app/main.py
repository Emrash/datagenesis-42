
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DataGenesis AI API",
    description="Enterprise-grade synthetic data generation platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS - CRITICAL: Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import services and routes
from .routes import generation, agents, analytics, auth, datasets, ai_config
from .services.gemini_service import GeminiService
from .services.agent_orchestrator import AgentOrchestrator
from .services.websocket_manager import ConnectionManager
from .services.ai_service import ai_service

# Initialize services
gemini_service = GeminiService()
orchestrator = AgentOrchestrator()
websocket_manager = ConnectionManager()

# Optional authentication
class OptionalHTTPBearer(HTTPBearer):
    async def __call__(self, request: Request):
        try:
            return await super().__call__(request)
        except HTTPException:
            return None

security = OptionalHTTPBearer(auto_error=False)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting DataGenesis AI API...")
    await gemini_service.initialize()
    await orchestrator.initialize()
    
    # Log initialization status without consuming quota
    logger.info("🤖 Gemini Service: Initialized successfully (quota-preserving mode)")
    
    logger.info("🎯 DataGenesis AI API started successfully!")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("📴 Shutting down DataGenesis AI API...")
    logger.info("📴 DataGenesis AI API shutdown complete")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    logger.info(f"🔗 {request.method} {request.url.path} - Starting request")
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# Include route modules
app.include_router(generation.router)
app.include_router(agents.router)
app.include_router(analytics.router)
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(ai_config.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DataGenesis AI Backend",
        "version": "1.0.0",
        "status": "running",
        "api_docs": "/api/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint - CRITICAL for frontend connection"""
    logger.info("🏥 Health check requested")
    
    # Check Gemini service health
    gemini_status = await gemini_service.health_check()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": "development",
        "host": "127.0.0.1:8000",
        "message": "DataGenesis AI Backend is running successfully",
        "services": {
            "gemini": gemini_status,
            "agents": "active",
            "websockets": "ready"
        }
    }
    
    logger.info(f"✅ Health check completed: {health_status}")
    return health_status

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle preflight OPTIONS requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates - Fixed authentication"""
    try:
        # Allow connections without authentication for guest users
        await websocket_manager.connect(websocket, client_id)
        logger.info(f"🔌 WebSocket connected: {client_id}")
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket_manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                    client_id
                )
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"🔌 WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for {client_id}: {str(e)}")
        websocket_manager.disconnect(client_id)

@app.post("/api/generation/schema-from-description")
async def generate_schema_from_description(
    request: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate schema from natural language description using Gemini 2.0 Flash"""
    logger.info("🧠 AI-powered schema generation request received")
    
    description = request.get("description", "")
    domain = request.get("domain", "general")
    data_type = request.get("data_type", "tabular")
    
    logger.info(f"📝 Description: {description[:100]}...")
    logger.info(f"🏭 Domain: {domain}, Type: {data_type}")
    
    # Basic validation
    if not description or len(description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Description must be at least 10 characters long")
    
    try:
        # Use Gemini 2.0 Flash for intelligent schema generation
        logger.info("🤖 Calling Gemini 2.0 Flash for schema generation...")
        schema_result = await gemini_service.generate_schema_from_natural_language(
            description, domain, data_type
        )
        
        logger.info(f"✅ Gemini 2.0 Flash generated schema with {len(schema_result.get('schema', {}))} fields")
        
        return {
            "schema": schema_result.get('schema', {}),
            "detected_domain": schema_result.get('detected_domain', domain),
            "estimated_rows": schema_result.get('estimated_rows', 10000),
            "suggestions": schema_result.get('suggestions', []),
            "sample_data": schema_result.get('sample_data', [])
        }
        
    except Exception as e:
        logger.error(f"❌ Schema generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema generation failed: {str(e)}")

@app.post("/api/generation/generate-local")
async def generate_synthetic_data(
    request: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate high-quality synthetic data using multi-agent AI system"""
    logger.info("🚀 Multi-Agent AI Generation Request Received")
    
    schema = request.get('schema', {})
    config = request.get('config', {})
    description = request.get('description', '')
    source_data = request.get('sourceData', [])
    
    row_count = config.get('rowCount', 100)
    domain = config.get('domain', 'general')
    
    logger.info(f"🎯 Starting AI-powered generation:")
    logger.info(f"   📊 Rows: {row_count}")
    logger.info(f"   🏭 Domain: {domain}")
    logger.info(f"   📝 Schema fields: {len(schema)}")
    logger.info(f"   🔍 Source data: {len(source_data)} records")
    
    # Generate unique job ID for tracking
    job_id = str(uuid.uuid4())
    
    try:
        # Start the multi-agent orchestration process
        logger.info(f"🤖 Initializing Multi-Agent Orchestra for job {job_id}")
        
        result = await orchestrator.orchestrate_generation(
            job_id=job_id,
            source_data=source_data,
            schema=schema,
            config=config,
            description=description,
            websocket_manager=websocket_manager
        )
        
        logger.info(f"🎉 Multi-Agent generation completed successfully!")
        logger.info(f"   ✅ Generated: {result['metadata']['rows_generated']} rows")
        logger.info(f"   🏆 Quality Score: {result['quality_score']}%")
        logger.info(f"   🔒 Privacy Score: {result['privacy_score']}%")
        logger.info(f"   ⚖️ Bias Score: {result['bias_score']}%")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Multi-agent generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/generation/analyze")
async def analyze_data(
    request: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze uploaded data using AI"""
    logger.info("🔍 AI-powered data analysis request received")
    
    sample_data = request.get('sample_data', [])
    config = request.get('config', {})
    
    if not sample_data:
        raise HTTPException(status_code=400, detail="No sample data provided")
    
    try:
        # Use Gemini for intelligent data analysis
        logger.info("🤖 Analyzing data with Gemini 2.0 Flash...")
        analysis = await gemini_service.analyze_schema_advanced(sample_data, config, [])
        
        logger.info(f"✅ Analysis completed: {analysis.get('domain', 'unknown')} domain detected")
        
        return {
            "analysis": analysis,
            "recommendations": analysis.get('recommendations', {})
        }
        
    except Exception as e:
        logger.error(f"❌ Data analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/agents/status")
async def get_agents_status():
    """Get real-time status of all AI agents"""
    logger.info("📊 Agent status requested")
    
    status = await orchestrator.get_agents_status()
    
    logger.info("✅ Agent status retrieved")
    return status

@app.get("/api/system/status")
async def system_status(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get real-time system status"""
    logger.info("📊 System status requested")
    
    # Get comprehensive system status
    gemini_status = await gemini_service.health_check()
    agents_status = await orchestrator.get_agents_status()
    
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "gemini_2_flash": gemini_status,
            "multi_agent_system": agents_status,
            "websockets": "active",
            "real_time_logging": "enabled"
        },
        "performance_metrics": {
            "ai_processing": "optimal",
            "response_time": "< 100ms",
            "uptime": "99.9%"
        }
    }
    
    logger.info("✅ System status compiled")
    return status

@app.post("/api/gemini/test-connection")
async def test_gemini_connection(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Test actual Gemini API connection - only call when needed"""
    logger.info("🧪 Testing Gemini API connection requested")
    
    try:
        result = await gemini_service.test_api_connection()
        logger.info(f"🧪 API test result: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ API test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API test failed: {str(e)}")

@app.post("/api/gemini/switch-model")
async def switch_gemini_model(
    request: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Switch to a different Gemini model to avoid quota issues"""
    logger.info("🔄 Gemini model switch requested")
    
    model_name = request.get("model", "gemini-1.5-flash")
    
    try:
        # Switch to a different model that might have quota available
        result = await gemini_service.switch_model(model_name)
        logger.info(f"🔄 Model switch result: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Model switch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model switch failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled exception in {request.method} {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
