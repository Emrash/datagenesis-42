"""
Multi-provider AI service for DataGenesis
Supports Gemini, OpenAI, Anthropic, and Ollama
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import google.generativeai as genai
from ..config import settings

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.current_provider = None
        self.current_model = None
        self.api_key = None
        self.endpoint = None
        self.is_initialized = False
        
    async def configure(self, provider: str, model: str, api_key: str, endpoint: str = None):
        """Configure the AI service with provider-specific settings"""
        self.current_provider = provider
        self.current_model = model
        self.api_key = api_key
        self.endpoint = endpoint
        
        try:
            if provider == 'gemini':
                await self._configure_gemini()
            elif provider == 'openai':
                await self._configure_openai()
            elif provider == 'anthropic':
                await self._configure_anthropic()
            elif provider == 'ollama':
                await self._configure_ollama()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            self.is_initialized = True
            logger.info(f"✅ AI Service configured for {provider} with model {model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to configure {provider}: {str(e)}")
            self.is_initialized = False
            return False
    
    async def _configure_gemini(self):
        """Configure Google Gemini"""
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.current_model)
        
    async def _configure_openai(self):
        """Configure OpenAI"""
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://api.openai.com/v1'
        
    async def _configure_anthropic(self):
        """Configure Anthropic Claude"""
        self.headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        self.base_url = 'https://api.anthropic.com/v1'
        
    async def _configure_ollama(self):
        """Configure Ollama"""
        self.base_url = self.endpoint or 'http://localhost:11434'
        self.headers = {'Content-Type': 'application/json'}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check AI service health"""
        if not self.is_initialized:
            return {
                "status": "error",
                "message": "Service not configured",
                "provider": self.current_provider,
                "model": self.current_model
            }
        
        try:
            if self.current_provider == 'gemini':
                return await self._health_check_gemini()
            elif self.current_provider == 'openai':
                return await self._health_check_openai()
            elif self.current_provider == 'anthropic':
                return await self._health_check_anthropic()
            elif self.current_provider == 'ollama':
                return await self._health_check_ollama()
                
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "provider": self.current_provider,
                "model": self.current_model
            }
    
    async def _health_check_gemini(self):
        """Gemini health check"""
        try:
            response = self.client.generate_content("test")
            return {
                "status": "online",
                "provider": "gemini",
                "model": self.current_model,
                "message": "Connection successful"
            }
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                return {
                    "status": "quota_exceeded",
                    "provider": "gemini", 
                    "model": self.current_model,
                    "message": f"Quota exceeded: {str(e)}"
                }
            raise e
    
    async def _health_check_openai(self):
        """OpenAI health check"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/models",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    return {
                        "status": "online",
                        "provider": "openai",
                        "model": self.current_model,
                        "message": "Connection successful"
                    }
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
    
    async def _health_check_anthropic(self):
        """Anthropic health check"""
        # Anthropic doesn't have a simple health check endpoint
        # We'll return optimistic status if configured
        return {
            "status": "ready",
            "provider": "anthropic",
            "model": self.current_model,
            "message": "Configured and ready"
        }
    
    async def _health_check_ollama(self):
        """Ollama health check"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    return {
                        "status": "online",
                        "provider": "ollama",
                        "model": self.current_model,
                        "message": "Connection successful",
                        "available_models": models
                    }
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
    
    async def generate_schema_from_natural_language(
        self,
        description: str,
        domain: str = 'general',
        data_type: str = 'tabular'
    ) -> Dict[str, Any]:
        """Generate schema from natural language description"""
        if not self.is_initialized:
            raise Exception("AI service not configured")
        
        prompt = self._build_schema_prompt(description, domain, data_type)
        
        if self.current_provider == 'gemini':
            return await self._generate_schema_gemini(prompt)
        elif self.current_provider == 'openai':
            return await self._generate_schema_openai(prompt)
        elif self.current_provider == 'anthropic':
            return await self._generate_schema_anthropic(prompt)
        elif self.current_provider == 'ollama':
            return await self._generate_schema_ollama(prompt)
        else:
            raise Exception(f"Schema generation not supported for {self.current_provider}")
    
    def _build_schema_prompt(self, description: str, domain: str, data_type: str) -> str:
        """Build schema generation prompt"""
        return f"""
        Based on this natural language description, generate a detailed database schema:
        
        Description: "{description}"
        Domain: {domain}
        Data Type: {data_type}
        
        Please create a comprehensive schema with:
        1. Realistic field names that match the described data
        2. Appropriate data types (string, number, boolean, date, email, phone, etc.)
        3. Constraints where applicable (min/max values, required fields)
        4. Sample values or examples for each field
        5. Domain-specific field suggestions
        
        Return the response as JSON with this exact structure:
        {{
            "schema": {{
                "field_name": {{
                    "type": "string|number|boolean|date|datetime|email|phone|uuid|text",
                    "description": "Clear description of the field",
                    "constraints": {{
                        "min": number,
                        "max": number,
                        "required": boolean,
                        "unique": boolean
                    }},
                    "examples": ["example1", "example2", "example3"]
                }}
            }},
            "detected_domain": "detected_domain_from_description",
            "estimated_rows": number,
            "relationships": ["description of data relationships"],
            "suggestions": ["suggestions for data generation"]
        }}
        """
    
    async def _generate_schema_gemini(self, prompt: str) -> Dict[str, Any]:
        """Generate schema using Gemini"""
        response = self.client.generate_content(prompt)
        return self._parse_json_response(response.text)
    
    async def _generate_schema_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate schema using OpenAI"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.current_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['choices'][0]['message']['content']
                    return self._parse_json_response(text)
                else:
                    raise Exception(f"OpenAI API error: {response.status}")
    
    async def _generate_schema_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Generate schema using Anthropic Claude"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.current_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000
            }
            
            async with session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['content'][0]['text']
                    return self._parse_json_response(text)
                else:
                    raise Exception(f"Anthropic API error: {response.status}")
    
    async def _generate_schema_ollama(self, prompt: str) -> Dict[str, Any]:
        """Generate schema using Ollama"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "stream": False
            }
            
            async with session.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['response']
                    return self._parse_json_response(text)
                else:
                    raise Exception(f"Ollama API error: {response.status}")
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from AI models"""
        try:
            # Clean up the response
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1]
            
            text = text.strip()
            parsed = json.loads(text)
            
            return {
                'schema': parsed.get('schema', {}),
                'detected_domain': parsed.get('detected_domain', 'general'),
                'estimated_rows': parsed.get('estimated_rows', 10000),
                'relationships': parsed.get('relationships', []),
                'suggestions': parsed.get('suggestions', [])
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response text: {text}")
            raise Exception("Invalid JSON response from AI model")

# Global AI service instance
ai_service = AIService()