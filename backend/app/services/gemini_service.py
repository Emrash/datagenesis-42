import google.generativeai as genai
import os
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.model = None
        self.is_initialized = False
        # Try multiple ways to get the API key
        self.api_key = (
            os.getenv('GEMINI_API_KEY') or 
            settings.gemini_api_key or 
            None
        )
        # Cache health status to avoid quota consumption
        self._last_health_check = None
        self._health_check_cache_ttl = 300  # 5 minutes
        self._last_health_check_time = 0
        
    async def initialize(self):
        """Initialize Gemini service"""
        try:
            if not self.api_key:
                logger.error("❌ GEMINI_API_KEY not found in environment variables or settings")
                logger.error("❌ Checked: os.getenv('GEMINI_API_KEY') and settings.gemini_api_key")
                return False
                
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.is_initialized = True
            logger.info("✅ Gemini 2.0 Flash initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
            self.is_initialized = False
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Gemini service health with caching to avoid quota consumption"""
        current_time = datetime.utcnow().timestamp()
        
        # Return cached result if available and not expired
        if (self._last_health_check and 
            current_time - self._last_health_check_time < self._health_check_cache_ttl):
            logger.info("🔄 Returning cached health status to save quota")
            return self._last_health_check
        
        if not self.api_key:
            result = {
                "status": "error",
                "model": "gemini-2.0-flash-exp",
                "message": "API key not configured",
                "api_key_configured": False,
                "api_key_status": "missing"
            }
            self._cache_health_result(result, current_time)
            return result
        
        if not self.is_initialized:
            result = {
                "status": "error", 
                "model": "gemini-2.0-flash-exp",
                "message": "Service not initialized",
                "api_key_configured": True,
                "api_key_status": "configured"
            }
            self._cache_health_result(result, current_time)
            return result
            
        # For initialized service, return optimistic status without API call
        # Only do actual API test if specifically requested
        result = {
            "status": "ready",
            "model": "gemini-2.0-flash-exp", 
            "message": "Initialized and ready (quota-preserving mode)",
            "api_key_configured": True,
            "api_key_status": "configured",
            "quota_preserved": True
        }
        
        self._cache_health_result(result, current_time)
        return result
    
    def _cache_health_result(self, result: Dict[str, Any], timestamp: float):
        """Cache health check result"""
        self._last_health_check = result
        self._last_health_check_time = timestamp
    
    async def test_api_connection(self) -> Dict[str, Any]:
        """Actually test API connection - only call when needed"""
        if not self.is_initialized:
            return {
                "status": "error",
                "message": "Service not initialized"
            }
            
        try:
            # This is the only method that should actually consume quota
            logger.info("🧪 Testing Gemini API connection (consuming quota)")
            response = self.model.generate_content("Say 'test'")
            return {
                "status": "online",
                "model": "gemini-2.0-flash-exp", 
                "message": "API connection successful",
                "api_test": "passed"
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "status": "quota_exceeded",
                    "model": "gemini-2.0-flash-exp",
                    "message": f"Quota exceeded: {error_msg}",
                    "api_test": "failed"
                }
            else:
                return {
                    "status": "error",
                    "model": "gemini-2.0-flash-exp", 
                    "message": f"Connection error: {error_msg}",
                    "api_test": "failed"
                }

    async def generate_synthetic_data(
        self,
        schema: Dict[str, Any],
        config: Dict[str, Any],
        description: str = "",
        source_data: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic data using Gemini"""
        logger.info("🤖 Generating synthetic data with Gemini 2.0 Flash...")

        if not self.is_initialized:
            raise Exception("Gemini service not initialized")

        # Handle both 'row_count' and 'rowCount'
        row_count = config.get('row_count') or config.get('rowCount') or 100

        prompt = f"""
        Generate {row_count} rows of realistic synthetic data based on this schema:

        Schema: {json.dumps(schema, indent=2)}
        Description: "{description}"
        Configuration: {json.dumps(config, indent=2)}

        Generate data that:
        1. Follows the exact schema structure
        2. Uses realistic values for each field type
        3. Maintains data relationships and constraints
        4. Ensures variety and realistic distribution
        5. Follows domain-specific patterns when applicable

        ⚠️ Return ONLY a JSON array of {row_count} objects, without any explanation, markdown or code fences.
        The output must be a valid JSON array where each element is an object matching the schema.
        Example of valid output format: [{{"field1": "value1", "field2": 123}}, {{"field1": "value2", "field2": 456}}]
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            logger.info(f"📝 Raw Gemini response length: {len(text)} characters")
            logger.debug(f"📝 Raw Gemini response (first 500 chars): {text[:500]}...")

            # Enhanced JSON extraction
            def extract_json(text: str) -> str:
                """Extract JSON from text with multiple fallback strategies"""
                # Try to find JSON in markdown code blocks first
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    blocks = text.split('```')
                    for block in blocks:
                        block = block.strip()
                        if (block.startswith('[') and block.endswith(']')) or (block.startswith('{') and block.endswith('}')):
                            text = block
                            break
                
                # Find the first valid JSON structure
                for start_char in ['[', '{']:
                    start_idx = text.find(start_char)
                    if start_idx != -1:
                        try:
                            # Find matching closing character
                            stack = []
                            end_idx = -1
                            for i in range(start_idx, len(text)):
                                char = text[i]
                                if char == start_char:
                                    stack.append(char)
                                elif (start_char == '[' and char == ']') or (start_char == '{' and char == '}'):
                                    stack.pop()
                                    if not stack:
                                        end_idx = i + 1
                                        break
                            if end_idx != -1:
                                candidate = text[start_idx:end_idx]
                                json.loads(candidate)  # Validate
                                return candidate
                        except json.JSONDecodeError:
                            continue
                
                # Last resort: find any substring that looks like JSON
                import re
                json_pattern = re.compile(r'(\[.*\]|\{.*\})', re.DOTALL)
                matches = json_pattern.findall(text)
                if matches:
                    for match in matches:
                        try:
                            json.loads(match)
                            return match
                        except json.JSONDecodeError:
                            continue
                
                raise ValueError("No valid JSON found in response")

            try:
                json_text = extract_json(text)
                logger.info(f"🧹 Extracted JSON length: {len(json_text)} characters")
                logger.debug(f"🧹 Extracted JSON: {json_text[:500]}...")

                data = json.loads(json_text)
                logger.info(f"✅ JSON parsed successfully - type: {type(data)}")

                # Normalize the data format to always return a list
                if isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        logger.info("🔄 Unwrapping 'data' key from response")
                        data = data['data']
                    elif all(isinstance(v, dict) for v in data.values()):
                        logger.info("🔄 Converting dict values to list")
                        data = list(data.values())
                    else:
                        logger.info("🔄 Converting single dict to list with one element")
                        data = [data]

                if not isinstance(data, list):
                    raise ValueError(f"Expected list after normalization, got {type(data)}")

                if len(data) > 0 and not isinstance(data[0], dict):
                    raise ValueError(f"Expected list of objects, got list of {type(data[0])}")

                logger.info(f"✅ Generated {len(data)} synthetic records with Gemini")
                return data[:row_count]

            except json.JSONDecodeError as json_err:
                logger.error(f"❌ JSON parsing failed: {str(json_err)}")
                logger.error(f"❌ Problematic JSON (first 500 chars): {text[:500]}...")
                raise ValueError(f"Failed to parse JSON response: {str(json_err)}")

        except Exception as e:
            logger.error(f"❌ Synthetic data generation failed: {str(e)}")
            raise Exception(f"Synthetic data generation failed: {str(e)}")

    async def analyze_schema_advanced(
        self,
        sample_data: List[Dict[str, Any]],
        config: Dict[str, Any],
        source_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Advanced schema analysis"""
        if not self.is_initialized:
            return {
                "domain": "general",
                "data_types": {},
                "relationships": [],
                "quality_score": 85,
                "pii_detected": False
            }

        prompt = f"""
        Analyze this dataset and provide comprehensive insights:
        Sample Data: {json.dumps(sample_data[:5], indent=2)}
        
        Provide analysis including:
        1. Detected domain (healthcare, finance, retail, etc.)
        2. Data types for each field
        3. Potential relationships between fields
        4. Quality assessment
        5. PII detection
        6. Suggestions for improvement
        
        Return as JSON with structure:
        {{
            "domain": "detected_domain",
            "data_types": {{}},
            "relationships": [],
            "quality_score": number,
            "pii_detected": boolean,
            "suggestions": []
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1]
            
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"❌ Schema analysis failed: {str(e)}")
            return {
                "domain": "general",
                "data_types": {},
                "relationships": [],
                "quality_score": 85,
                "pii_detected": False,
                "error": str(e)
            }

    async def assess_privacy_risks(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess privacy risks in data"""
        if not self.is_initialized:
            return {
                "privacy_score": 90,
                "pii_detected": [],
                "sensitive_attributes": [],
                "risk_level": "low",
                "recommendations": ["Gemini service not available for privacy analysis"]
            }

        prompt = f"""
        Assess privacy risks in this dataset:
        Data Sample: {json.dumps(data[:3], indent=2)}
        
        Check for:
        1. PII (Personally Identifiable Information)
        2. Sensitive attributes
        3. Re-identification risks
        4. Data linkage possibilities
        
        Return as JSON:
        {{
            "privacy_score": number_0_to_100,
            "pii_detected": ["list of detected PII fields"],
            "sensitive_attributes": ["list of sensitive fields"],
            "risk_level": "low|medium|high",
            "recommendations": ["privacy improvement suggestions"]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1]
            
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"❌ Privacy assessment failed: {str(e)}")
            return {
                "privacy_score": 85,
                "pii_detected": [],
                "sensitive_attributes": [],
                "risk_level": "medium",
                "recommendations": [f"Privacy analysis error: {str(e)}"]
            }

    async def analyze_data_comprehensive(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive data analysis for domain detection"""
        if not self.is_initialized:
            return {
                "domain": "general",
                "confidence": 0.7,
                "data_quality": {"score": 85, "issues": [], "recommendations": []},
                "schema_inference": {},
                "recommendations": {"generation_strategy": "Standard generation - AI analysis unavailable"}
            }

        prompt = f"""
        Analyze this dataset comprehensively and determine its domain:
        Sample Data: {json.dumps(data[:3], indent=2)}
        
        Provide detailed analysis including:
        1. Domain classification (healthcare, finance, retail, manufacturing, education, etc.)
        2. Confidence level (0-1)
        3. Data quality assessment
        4. Schema inference for each field
        5. Recommendations for synthetic data generation
        
        Return as JSON:
        {{
            "domain": "detected_domain",
            "confidence": 0.9,
            "data_quality": {{
                "score": 85,
                "issues": ["list of issues"],
                "recommendations": ["list of recommendations"]
            }},
            "schema_inference": {{
                "field_name": "inferred_type_and_pattern"
            }},
            "recommendations": {{
                "generation_strategy": "specific strategy for this domain"
            }}
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1]
            
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
            return {
                "domain": "general",
                "confidence": 0.7,
                "data_quality": {"score": 85, "issues": [], "recommendations": []},
                "schema_inference": {},
                "recommendations": {"generation_strategy": "Standard generation - AI analysis unavailable"},
                "error": str(e)
            }

    async def detect_bias_comprehensive(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive bias detection and analysis"""
        if not self.is_initialized:
            return {
                "bias_score": 88,
                "detected_biases": [],
                "bias_types": [],
                "recommendations": ["Enable Gemini API for advanced bias detection"]
            }

        prompt = f"""
        Analyze this dataset for potential biases:
        Sample Data: {json.dumps(data[:3], indent=2)}
        
        Check for:
        1. Selection bias
        2. Demographic bias (gender, age, race)
        3. Geographic bias
        4. Temporal bias
        5. Representation bias
        
        Return as JSON:
        {{
            "bias_score": number_0_to_100,
            "detected_biases": ["list of detected biases"],
            "bias_types": ["selection", "demographic", "geographic", "temporal"],
            "severity": "low|medium|high",
            "recommendations": ["specific mitigation strategies"],
            "mitigation_strategies": ["actionable steps"]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1]
            
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"❌ Bias detection failed: {str(e)}")
            return {
                "bias_score": 88,
                "detected_biases": [],
                "bias_types": [],
                "recommendations": [f"Bias detection error: {str(e)}"],
                "mitigation_strategies": []
            }

    async def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different Gemini model to avoid quota issues"""
        logger.info(f"🔄 Switching to model: {model_name}")
        
        if not self.api_key:
            return {
                "status": "error",
                "message": "API key not configured",
                "current_model": None,
                "new_model": model_name
            }
        
        try:
            # List of available models to try
            available_models = [
                "gemini-1.5-flash",
                "gemini-1.5-pro", 
                "gemini-2.0-flash-exp",
                "gemini-1.0-pro"
            ]
            
            if model_name not in available_models:
                return {
                    "status": "error",
                    "message": f"Model {model_name} not available. Available models: {available_models}",
                    "current_model": getattr(self.model, 'model_name', None),
                    "new_model": model_name
                }
            
            # Configure new model
            old_model = getattr(self.model, 'model_name', 'unknown') if self.model else 'none'
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            
            # Clear health check cache to force revalidation
            self._last_health_check = None
            self._last_health_check_time = 0
            
            logger.info(f"✅ Successfully switched from {old_model} to {model_name}")
            
            return {
                "status": "success",
                "message": f"Successfully switched to {model_name}",
                "previous_model": old_model,
                "current_model": model_name,
                "available_models": available_models
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to switch model: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to switch model: {str(e)}",
                "current_model": getattr(self.model, 'model_name', None),
                "new_model": model_name
            }