
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database
    supabase_url: str = "https://yrwcudnujriyppmpxtko.supabase.co"
    supabase_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyd2N1ZG51anJpeXBwbXB4dGtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE2NDQ4NzgsImV4cCI6MjA2NzIyMDg3OH0.qKktoZ_cg0zUbPtJiLlindE4iNUExPp3txtZrhOP9SY" 
    supabase_service_role_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyd2N1ZG51anJpeXBwbXB4dGtvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTY0NDg3OCwiZXhwIjoyMDY3MjIwODc4fQ.oPHhxSMmV2CaZm_RpmY-A06FqRCcaf4iQ2Py1dosYt8"
    
    # Redis
    redis_url: str = "redis://default:2kjMvjplKbYLVZSrNftrWfFfC6bGNak9@redis-13890.c16.us-east-1-3.ec2.redns.redis-cloud.com:13890"
    redis_password: Optional[str] = "2kjMvjplKbYLVZSrNftrWfFfC6bGNak9"
    
    # AI Services - Fixed to properly read from .env
    gemini_api_key: str = "AIzaSyA81SV6mvA9ShZasJgcVl4ps-YQm9DrKsc"
    google_cloud_project_id: Optional[str] = "gen-lang-client-0626319060"
    
    # Vector Database
    pinecone_api_key: str = "pcsk_5bsEtw_SwNGTxmapgiejijc9sYQ6X3ygsToxgzeutJ2Rj3xVpbApPbueQ9n1eNKYY8Di23"
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "databank"
    
    # Security
    secret_key: str = "databank@super_secure_key115"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Celery/Background Jobs
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # API Settings
    api_v1_str: str = "/api"
    project_name: str = "DataGenesis AI"
    
    # Generation Settings
    max_concurrent_generations: int = 5
    max_dataset_size_mb: int = 100
    default_cache_ttl: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Add extra configuration to help with environment loading
        extra = "allow"
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure we get the GEMINI_API_KEY if it's not loaded properly
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDOFWw2dk2W28l52mXJxJXrcdYbxsQz13s')

settings = Settings()
