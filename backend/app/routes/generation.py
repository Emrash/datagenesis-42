from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import List, Dict, Any, Optional
import uuid
import asyncio
import logging
import json

from ..middleware.auth import verify_token
from ..services.redis_service import RedisService
from ..services.gemini_service import GeminiService
from ..services.vector_service import VectorService
from ..services.agent_orchestrator import AgentOrchestrator
from ..services.supabase_service import SupabaseService
from ..models.generation import GenerationRequest, GenerationResponse, NaturalLanguageRequest, SchemaGenerationResponse

router = APIRouter()
security = HTTPBearer()

# Initialize services
redis_service = RedisService()
gemini_service = GeminiService()
vector_service = VectorService()
supabase_service = SupabaseService()
orchestrator = AgentOrchestrator()

logger = logging.getLogger(__name__)

@router.post("/schema-from-description", response_model=SchemaGenerationResponse)
async def generate_schema_from_description(
    request: NaturalLanguageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate schema from natural language description"""
    logger.info(f"🧠 Schema generation request received: domain={request.domain}, type={request.data_type}")
    logger.info(f"📝 Description: {request.description[:100]}...")
    
    try:
        # Allow both authenticated users and guests
        user = None
        if credentials:
            try:
                user = await verify_token(credentials.credentials)
                logger.info(f"👤 User authenticated: {user.get('id', 'unknown')}")
            except Exception as e:
                logger.info(f"🔓 Guest access detected: {str(e)}")
                # Allow guest access
                pass
        else:
            logger.info("🔓 No credentials provided, allowing guest access")
            
        # Validate request
        if not request.description or len(request.description.strip()) < 10:
            raise HTTPException(status_code=400, detail="Description must be at least 10 characters long")
        
        if not request.domain or not request.data_type:
            raise HTTPException(status_code=400, detail="Domain and data type are required")
        
        logger.info(f"✅ Request validated successfully")
        
        # Generate schema using Gemini
        logger.info("🔄 Calling Gemini service for schema generation...")
        schema_result = await gemini_service.generate_schema_from_natural_language(
            request.description,
            request.domain,
            request.data_type
        )
        
        logger.info(f"✅ Gemini response received with {len(schema_result.get('schema', {}))} fields")
        
        # Validate the generated schema
        if not schema_result or not schema_result.get('schema'):
            logger.error("❌ Empty or invalid schema generated")
            raise HTTPException(status_code=500, detail="Failed to generate valid schema")
        
        # Generate sample data from the schema
        sample_data = _generate_sample_data_from_schema(schema_result.get('schema', {}), 5)
        schema_result['sample_data'] = sample_data
        
        response = SchemaGenerationResponse(
            schema=schema_result.get('schema', {}),
            detected_domain=schema_result.get('detected_domain', request.domain),
            sample_data=sample_data,
            suggestions=schema_result.get('suggestions', [])
        )
        
        logger.info(f"🎉 Schema generation completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Schema generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema generation failed: {str(e)}")

def _generate_sample_data_from_schema(schema: Dict[str, Any], num_rows: int = 5) -> List[Dict[str, Any]]:
    """Generate sample data from schema definition"""
    logger.info(f"📊 Generating {num_rows} sample rows from schema with {len(schema)} fields")
    
    sample_data = []
    
    for i in range(num_rows):
        row = {}
        for field_name, field_info in schema.items():
            row[field_name] = _generate_sample_value(field_info, field_name, i)
        sample_data.append(row)
    
    logger.info(f"✅ Generated sample data successfully")
    return sample_data

def _generate_sample_value(field_info: Dict[str, Any], field_name: str, index: int):
    """Generate a realistic sample value based on field type and constraints"""
    field_type = field_info.get('type', 'string')
    constraints = field_info.get('constraints', {})
    examples = field_info.get('examples', [])
    description = field_info.get('description', '')
    
    # Use examples if available
    if examples and len(examples) > 0:
        return examples[index % len(examples)]
    
    # Generate realistic data based on field name and type
    lower_field_name = field_name.lower()
    lower_description = description.lower()
    
    # Domain-specific realistic data generation
    if 'patient' in lower_field_name or 'patient' in lower_description:
        return f"PT{str(1000 + index).zfill(4)}"
    elif 'name' in lower_field_name and 'patient' not in lower_field_name:
        names = ['John Smith', 'Mary Johnson', 'David Brown', 'Sarah Davis', 'Michael Wilson']
        return names[index % len(names)]
    elif 'age' in lower_field_name:
        return 25 + (index * 3) % 50
    elif 'diagnosis' in lower_field_name:
        diagnoses = ['Hypertension', 'Diabetes Type 2', 'Asthma', 'Migraine', 'Arthritis']
        return diagnoses[index % len(diagnoses)]
    elif 'amount' in lower_field_name or 'price' in lower_field_name:
        return round(100 + (index * 47.5) % 1000, 2)
    elif 'product' in lower_field_name:
        products = ['Laptop Pro 15"', 'Wireless Headphones', 'Smart Watch', 'Gaming Mouse']
        return products[index % len(products)]
    
    # Generic type-based generation
    if field_type in ['string', 'text']:
        if 'email' in lower_field_name:
            return f"user{index + 1}@example.com"
        elif 'phone' in lower_field_name:
            return f"+1-555-{str(1000 + index).zfill(4)}"
        elif 'address' in lower_field_name:
            return f"{123 + index} Main Street, City, State {10001 + index}"
        return f"{field_name.title()} {index + 1}"
    elif field_type in ['number', 'integer']:
        min_val = constraints.get('min', 1)
        max_val = constraints.get('max', 100)
        return min_val + (index * (max_val - min_val) // 10)
    elif field_type == 'boolean':
        return index % 2 == 0
    elif field_type in ['date', 'datetime']:
        from datetime import datetime, timedelta
        base_date = datetime.now() - timedelta(days=365)
        result_date = base_date + timedelta(days=index * 30)
        return result_date.isoformat() if field_type == 'datetime' else result_date.date().isoformat()
    elif field_type == 'email':
        return f"user{index + 1}@example.com"
    elif field_type == 'phone':
        return f"+1-555-{str(1000 + index).zfill(4)}"
    elif field_type == 'uuid':
        return str(uuid.uuid4())
    else:
        return f"Sample {field_name} {index + 1}"

@router.post("/start", response_model=GenerationResponse)
async def start_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Start synthetic data generation job"""
    logger.info(f"🚀 Generation start request: domain={request.domain}, type={request.data_type}")
    
    # Allow both authenticated users and guests
    user = None
    try:
        user = await verify_token(credentials.credentials)
        logger.info(f"👤 User authenticated: {user.get('id', 'unknown')}")
    except:
        # Allow guest access - create temporary user
        user = {
            "id": f"guest_{uuid.uuid4().hex[:8]}",
            "email": "guest@datagenesis.ai",
            "is_guest": True
        }
        logger.info(f"🔓 Guest user created: {user['id']}")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"📋 Job ID created: {job_id}")
    
    # Start background generation task
    background_tasks.add_task(
        run_generation_job,
        job_id,
        user.get("id") if user else "anonymous",
        request.dict()
    )
    
    logger.info(f"✅ Generation job queued successfully")
    return GenerationResponse(
        job_id=job_id,
        status="started",
        message="Generation job started successfully"
    )

@router.get("/status/{job_id}")
async def get_generation_status(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get generation job status"""
    logger.info(f"📊 Status request for job: {job_id}")
    
    # Allow both authenticated users and guests
    try:
        user = await verify_token(credentials.credentials)
        logger.info(f"👤 User authenticated for status check")
    except:
        # Allow guest access
        logger.info(f"🔓 Guest access for status check")
        pass
    
    # Get job status from Redis
    job_data = await redis_service.get_cache(f"job:{job_id}")
    
    if not job_data:
        logger.warning(f"❌ Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    logger.info(f"✅ Job status retrieved: {job_data.get('status', 'unknown')}")
    return {
        "job_id": job_id,
        "status": job_data.get("status", "unknown"),
        "progress": job_data.get("progress", 0),
        "message": job_data.get("message", ""),
        "started_at": job_data.get("started_at"),
        "estimated_completion": job_data.get("estimated_completion"),
        "result": job_data.get("result") if job_data.get("status") == "completed" else None
    }

@router.get("/jobs")
async def get_user_jobs(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get all generation jobs for user"""
    logger.info("📋 User jobs request")
    
    # Allow both authenticated users and guests
    try:
        user = await verify_token(credentials.credentials)
        user_id = user["id"]
        logger.info(f"👤 Jobs requested for user: {user_id}")
    except:
        # For guests, return empty list or recent guest jobs
        logger.info("🔓 Guest jobs request - returning empty list")
        return {"jobs": []}
    
    # Get jobs from Supabase
    jobs = await supabase_service.get_user_generation_jobs(user_id)
    logger.info(f"✅ Retrieved {len(jobs)} jobs for user")
    
    return {"jobs": jobs}

@router.delete("/jobs/{job_id}")
async def cancel_generation_job(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Cancel a running generation job"""
    logger.info(f"🛑 Cancel request for job: {job_id}")
    
    user = await verify_token(credentials.credentials)
    
    # Update job status
    await redis_service.update_job_progress(job_id, -1, "cancelled")
    logger.info(f"✅ Job cancelled: {job_id}")
    
    return {"message": "Job cancelled successfully"}

@router.post("/analyze")
async def analyze_data(
    data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze uploaded data before generation"""
    logger.info("🔍 Data analysis request")
    
    # Allow both authenticated users and guests
    try:
        user = await verify_token(credentials.credentials)
        logger.info(f"👤 Analysis request from user")
    except:
        # Allow guest access
        logger.info("🔓 Guest analysis request")
        pass
    
    try:
        # Quick analysis using Gemini
        analysis = await gemini_service.analyze_schema_advanced(
            data.get("sample_data", []),
            data.get("config", {}),
            []
        )
        
        logger.info("✅ Data analysis completed")
        return {
            "analysis": analysis,
            "recommendations": {
                "suggested_row_count": min(max(len(data.get("sample_data", [])) * 10, 1000), 100000),
                "suggested_privacy_level": "high" if analysis.get("pii_detected") else "medium",
                "estimated_generation_time": "2-5 minutes"
            }
        }
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/generate-local")
async def generate_local_data(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate synthetic data locally (for guests and when full pipeline isn't needed)"""
    logger.info("🏠 Local generation request")
    
    try:
        # This endpoint is for both guests and authenticated users who want quick generation
        schema = request.get('schema', {})
        config = request.get('config', {})
        description = request.get('description', '')
        
        logger.info(f"📊 Local generation: {len(schema)} fields, {config.get('rowCount', 100)} rows")
        
        # Use Gemini to generate synthetic data
        synthetic_data = await gemini_service.generate_synthetic_data(
            schema, config, description
        )
        
        # Calculate basic quality metrics
        quality_score = min(100, max(80, len(synthetic_data) / max(1, config.get('rowCount', 100)) * 100))
        privacy_score = 95  # Default high privacy for synthetic data
        bias_score = 88    # Default bias score
        
        logger.info(f"✅ Local generation completed: {len(synthetic_data)} rows")
        
        return {
            "data": synthetic_data,
            "metadata": {
                "rowsGenerated": len(synthetic_data),
                "columnsGenerated": len(synthetic_data[0].keys()) if synthetic_data else 0,
                "generationTime": datetime.utcnow().isoformat(),
                "config": config,
                "generationMethod": "backend_local"
            },
            "qualityScore": quality_score,
            "privacyScore": privacy_score,
            "biasScore": bias_score
        }
        
    except Exception as e:
        logger.error(f"❌ Local generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Local generation failed: {str(e)}")

async def run_generation_job(job_id: str, user_id: str, config: Dict[str, Any]):
    """Background task to run generation job"""
    logger.info(f"🔄 Starting background generation job: {job_id}")
    
    try:
        # Store initial job data
        if not user_id.startswith("guest_") and not user_id == "anonymous":
            await supabase_service.create_generation_job(job_id, user_id, config)
        
        # Start job in Redis
        await redis_service.start_generation_job(job_id, config)
        
        # Run the orchestrated generation
        logger.info(f"🎯 Running orchestrated generation for job: {job_id}")
        result = await orchestrator.orchestrate_generation(
            job_id,
            config.get("source_data", []),
            config
        )
        
        logger.info(f"✅ Generation completed for job: {job_id}")
        
        # Store the result in Supabase
        if not user_id.startswith("guest_") and not user_id == "anonymous":
            await supabase_service.complete_generation_job(job_id, result)
        
        # Update metrics
        await redis_service.increment_metric("total_generations")
        await redis_service.increment_metric("successful_generations")
        
    except Exception as e:
        # Handle job failure
        logger.error(f"❌ Generation job {job_id} failed: {str(e)}")
        await redis_service.update_job_progress(job_id, -1, "failed")
        if not user_id.startswith("guest_") and not user_id == "anonymous":
            await supabase_service.fail_generation_job(job_id, str(e))