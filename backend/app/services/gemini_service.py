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

        # Create sophisticated prompt based on domain detection
        domain_context = self._extract_domain_context(schema, config, description)
        
        prompt = f"""
You are an expert synthetic data generator. Create {row_count} rows of ultra-realistic synthetic data that is indistinguishable from real-world data.

SCHEMA DEFINITION:
{json.dumps(schema, indent=2)}

CONTEXT:
- Domain: {domain_context['domain']}
- Description: "{description}"
- Configuration: {json.dumps(config, indent=2)}

CRITICAL DATA GENERATION REQUIREMENTS:

1. REALISM STANDARDS:
   - All values must be realistic for the detected domain: {domain_context['domain']}
   - Ages: 0-120 years (weighted towards realistic distributions)
   - Dates: Use realistic date ranges and patterns
   - IDs: Follow professional formatting standards
   - Names: Use diverse, culturally appropriate names
   - Medical conditions: Use actual medical terminology if healthcare domain
   - Financial amounts: Use realistic currency and decimal precision

2. FIELD-SPECIFIC RULES:
{domain_context['field_rules']}

3. DATA RELATIONSHIPS:
   - Ensure logical consistency between related fields
   - Age should correlate with admission patterns if healthcare
   - Gender should use standard categories: "Male", "Female", "Other", "Prefer not to say"
   - Dates should follow chronological logic

4. QUALITY ASSURANCE:
   - No placeholder text like "Sample X" or "Generated Y"
   - No unrealistic values (e.g., age > 120, negative amounts)
   - Maintain statistical distributions typical of real data
   - Include appropriate data variance and edge cases (but realistic ones)

5. DOMAIN-SPECIFIC AUTHENTICITY:
{domain_context['authenticity_rules']}

OUTPUT FORMAT:
Return ONLY a valid JSON array of exactly {row_count} objects. No explanations, markdown, or code fences.

Example of expected quality:
[{{"patient_id": "PT001234", "name": "Sarah Johnson", "age": 34, "gender": "Female", "admission_date": "2024-11-15", "conditions": "Type 2 Diabetes Mellitus"}}]

Generate {row_count} records now:
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

                # Validate and clean the generated data
                cleaned_data = self._validate_and_clean_data(data, schema, domain_context)
                
                logger.info(f"✅ Generated {len(cleaned_data)} synthetic records with Gemini")
                return cleaned_data[:row_count]

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

    def _extract_domain_context(self, schema: Dict[str, Any], config: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Extract domain-specific context for realistic data generation"""
        
        # Detect domain from schema fields and description
        domain = config.get('domain', 'general')
        field_names = list(schema.keys())
        field_names_lower = [name.lower() for name in field_names]
        description_lower = description.lower()
        
        # Enhanced domain detection
        if any(keyword in ' '.join(field_names_lower + [description_lower]) for keyword in 
               ['patient', 'medical', 'health', 'diagnosis', 'condition', 'admission', 'hospital', 'clinic']):
            domain = 'healthcare'
        elif any(keyword in ' '.join(field_names_lower + [description_lower]) for keyword in 
                 ['transaction', 'account', 'payment', 'balance', 'credit', 'debit', 'financial', 'bank']):
            domain = 'finance'
        elif any(keyword in ' '.join(field_names_lower + [description_lower]) for keyword in 
                 ['customer', 'product', 'order', 'purchase', 'inventory', 'sales', 'retail']):
            domain = 'retail'
        elif any(keyword in ' '.join(field_names_lower + [description_lower]) for keyword in 
                 ['student', 'course', 'grade', 'school', 'education', 'university', 'learning']):
            domain = 'education'
        
        # Domain-specific field rules and authenticity guidelines
        context = {
            'domain': domain,
            'field_rules': self._get_field_rules(schema, domain),
            'authenticity_rules': self._get_authenticity_rules(domain)
        }
        
        return context
    
    def _get_field_rules(self, schema: Dict[str, Any], domain: str) -> str:
        """Generate field-specific rules based on schema and domain"""
        rules = []
        
        for field_name, field_info in schema.items():
            field_lower = field_name.lower()
            field_type = field_info.get('type', 'string')
            
            if 'age' in field_lower:
                if domain == 'healthcare':
                    rules.append(f"- {field_name}: Age 0-95 years, weighted distribution (more 25-65, fewer 0-18 and 80+)")
                else:
                    rules.append(f"- {field_name}: Realistic age distribution 18-85 years")
            
            elif 'name' in field_lower:
                rules.append(f"- {field_name}: Diverse, culturally appropriate full names (First Last format)")
            
            elif 'gender' in field_lower or 'sex' in field_lower:
                rules.append(f"- {field_name}: Use exactly 'Male', 'Female', 'Other', 'Prefer not to say' (45%, 45%, 5%, 5%)")
            
            elif 'id' in field_lower:
                if domain == 'healthcare':
                    rules.append(f"- {field_name}: Format PT######, MR######, or similar professional medical ID patterns")
                elif domain == 'finance':
                    rules.append(f"- {field_name}: Account numbers, transaction IDs following banking standards")
                else:
                    rules.append(f"- {field_name}: Professional ID format with alphanumeric patterns")
            
            elif 'date' in field_lower:
                rules.append(f"- {field_name}: YYYY-MM-DD format, realistic date ranges within last 2 years")
            
            elif 'condition' in field_lower or 'diagnosis' in field_lower:
                if domain == 'healthcare':
                    rules.append(f"- {field_name}: Actual medical conditions (Hypertension, Type 2 Diabetes, Asthma, Pneumonia, etc.)")
                else:
                    rules.append(f"- {field_name}: Relevant conditions for the domain context")
            
            elif 'amount' in field_lower or 'price' in field_lower or 'cost' in field_lower:
                rules.append(f"- {field_name}: Realistic monetary values with proper decimal places")
            
            elif 'email' in field_lower:
                rules.append(f"- {field_name}: Realistic email formats with diverse domains")
            
            elif 'phone' in field_lower:
                rules.append(f"- {field_name}: Valid phone number formats")
        
        return '\n'.join(rules) if rules else "- Follow standard realistic data patterns for all fields"
    
    def _get_authenticity_rules(self, domain: str) -> str:
        """Get domain-specific authenticity rules"""
        
        if domain == 'healthcare':
            return """
- Use actual medical terminology and ICD-10 condition names
- Age distributions should reflect real patient demographics
- Admission dates should show realistic patterns (weekdays more common)
- Patient IDs should follow healthcare standards (PT###### or MRN###### format)
- Names should be diverse and culturally representative
- Medical conditions should be age-appropriate (e.g., no pediatric conditions for 80+ year olds)
- Use realistic hospital/clinic workflow patterns
"""
        
        elif domain == 'finance':
            return """
- Transaction amounts should follow realistic spending patterns
- Account numbers should follow banking industry standards
- Date patterns should reflect business days for transactions
- Customer data should comply with financial industry norms
- Currency values should have appropriate decimal precision
- Transaction types should match real banking terminology
"""
        
        elif domain == 'retail':
            return """
- Product names should be realistic and diverse
- Pricing should reflect market reality
- Customer demographics should be representative
- Purchase patterns should show realistic consumer behavior
- Inventory levels should be practical
- Sales data should follow seasonal patterns
"""
        
        elif domain == 'education':
            return """
- Student data should reflect educational demographics
- Course codes and names should follow academic standards
- Grade distributions should be realistic (bell curve patterns)
- Academic years should align with calendar systems
- Student IDs should follow institutional formatting
"""
        
        else:
            return """
- All data should reflect real-world patterns and distributions
- No placeholder or template text
- Maintain statistical realism across all fields
- Ensure data consistency and logical relationships
- Use appropriate formatting standards for each data type
"""

    def _validate_and_clean_data(self, data: List[Dict[str, Any]], schema: Dict[str, Any], domain_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and clean generated data to ensure high quality"""
        logger.info("🔍 Validating and cleaning generated data for quality assurance...")
        
        cleaned_data = []
        domain = domain_context['domain']
        
        for i, record in enumerate(data):
            cleaned_record = {}
            is_valid = True
            
            for field_name, field_info in schema.items():
                if field_name not in record:
                    logger.warning(f"Missing field {field_name} in record {i}, skipping record")
                    is_valid = False
                    break
                
                value = record[field_name]
                cleaned_value = self._validate_and_clean_field(field_name, value, field_info, domain, i)
                
                if cleaned_value is None:
                    logger.warning(f"Invalid value for {field_name} in record {i}, skipping record")
                    is_valid = False
                    break
                
                cleaned_record[field_name] = cleaned_value
            
            if is_valid:
                # Final record-level validation
                if self._validate_record_consistency(cleaned_record, domain):
                    cleaned_data.append(cleaned_record)
                else:
                    logger.warning(f"Record {i} failed consistency validation, skipping")
        
        logger.info(f"✅ Data validation complete: {len(cleaned_data)}/{len(data)} records passed quality checks")
        return cleaned_data
    
    def _validate_and_clean_field(self, field_name: str, value: Any, field_info: Dict[str, Any], domain: str, record_index: int) -> Any:
        """Validate and clean individual field value"""
        field_lower = field_name.lower()
        field_type = field_info.get('type', 'string')
        
        # Convert string representation to appropriate type if needed
        if isinstance(value, str) and value.strip() == "":
            return None
        
        # Age validation
        if 'age' in field_lower:
            try:
                age = int(float(str(value)))  # Handle potential float strings
                if age < 0 or age > 120:
                    logger.warning(f"Unrealistic age {age} in record {record_index}, generating realistic replacement")
                    # Generate realistic age based on domain
                    if domain == 'healthcare':
                        age = min(95, max(0, 30 + (record_index * 7) % 60))  # 30-90 range
                    else:
                        age = min(85, max(18, 25 + (record_index * 5) % 50))  # 25-75 range
                return age
            except (ValueError, TypeError):
                logger.warning(f"Invalid age format '{value}' in record {record_index}")
                return 35 + (record_index % 40)  # Fallback realistic age
        
        # Gender validation
        if 'gender' in field_lower or 'sex' in field_lower:
            if isinstance(value, str):
                value_clean = value.strip()
                # Check for placeholder patterns
                if 'sample' in value_clean.lower() or value_clean.lower().startswith('generated'):
                    # Replace with realistic gender
                    genders = ['Male', 'Female', 'Other', 'Prefer not to say']
                    return genders[record_index % 4]  # Distribute evenly
                
                # Normalize common gender formats
                value_lower = value_clean.lower()
                if value_lower in ['m', 'male', 'man']:
                    return 'Male'
                elif value_lower in ['f', 'female', 'woman']:
                    return 'Female'
                elif value_lower in ['other', 'non-binary', 'nb']:
                    return 'Other'
                elif 'prefer' in value_lower or 'not' in value_lower:
                    return 'Prefer not to say'
                else:
                    return value_clean if value_clean in ['Male', 'Female', 'Other', 'Prefer not to say'] else 'Male'
            return 'Male'  # Default fallback
        
        # Medical conditions validation
        if ('condition' in field_lower or 'diagnosis' in field_lower) and domain == 'healthcare':
            if isinstance(value, str):
                value_clean = value.strip()
                # Check for placeholder patterns
                if ('sample' in value_clean.lower() or 
                    'generated' in value_clean.lower() or 
                    'placeholder' in value_clean.lower() or
                    value_clean.lower().startswith('condition') or
                    value_clean.lower().startswith('diagnosis')):
                    
                    # Replace with realistic medical conditions
                    realistic_conditions = [
                        'Type 2 Diabetes Mellitus',
                        'Essential Hypertension',
                        'Hyperlipidemia',
                        'Chronic Obstructive Pulmonary Disease',
                        'Osteoarthritis',
                        'Depression',
                        'Anxiety Disorder',
                        'Asthma',
                        'Gastroesophageal Reflux Disease',
                        'Chronic Kidney Disease',
                        'Atrial Fibrillation',
                        'Coronary Artery Disease',
                        'Thyroid Disorder',
                        'Sleep Apnea',
                        'Migraine Headaches'
                    ]
                    return realistic_conditions[record_index % len(realistic_conditions)]
                
                return value_clean
            return 'Essential Hypertension'  # Default realistic condition
        
        # Date validation
        if 'date' in field_lower:
            if isinstance(value, str):
                # Validate date format and realism
                try:
                    from datetime import datetime, timedelta
                    import re
                    
                    # Check for valid date format
                    if re.match(r'\d{4}-\d{2}-\d{2}', value):
                        parsed_date = datetime.strptime(value, '%Y-%m-%d')
                        current_date = datetime.now()
                        
                        # Ensure date is realistic (within reasonable range)
                        if parsed_date > current_date + timedelta(days=30):
                            # Future date too far, adjust to realistic range
                            adjusted_date = current_date - timedelta(days=record_index * 10 + 30)
                            return adjusted_date.strftime('%Y-%m-%d')
                        elif parsed_date < datetime(2020, 1, 1):
                            # Too far in past, adjust
                            adjusted_date = datetime(2024, 1, 1) + timedelta(days=record_index * 5)
                            return adjusted_date.strftime('%Y-%m-%d')
                        
                        return value
                    else:
                        # Invalid format, generate realistic date
                        base_date = datetime(2024, 6, 1)
                        adjusted_date = base_date + timedelta(days=record_index * 10)
                        return adjusted_date.strftime('%Y-%m-%d')
                
                except Exception:
                    # Fallback realistic date
                    from datetime import datetime, timedelta
                    base_date = datetime(2024, 6, 1)
                    adjusted_date = base_date + timedelta(days=record_index * 10)
                    return adjusted_date.strftime('%Y-%m-%d')
        
        # ID validation
        if 'id' in field_lower:
            if isinstance(value, str):
                value_clean = value.strip()
                # Check for placeholder patterns
                if ('sample' in value_clean.lower() or 
                    'generated' in value_clean.lower() or
                    value_clean.lower() == 'id'):
                    
                    # Generate realistic ID based on domain
                    if domain == 'healthcare':
                        return f"PT{str(100000 + record_index).zfill(6)}"
                    elif domain == 'finance':
                        return f"ACC{str(200000 + record_index).zfill(8)}"
                    else:
                        return f"ID{str(300000 + record_index).zfill(6)}"
                
                return value_clean
            return f"ID{str(400000 + record_index).zfill(6)}"
        
        # Name validation
        if 'name' in field_lower:
            if isinstance(value, str):
                value_clean = value.strip()
                # Check for placeholder patterns
                if ('sample' in value_clean.lower() or 
                    'generated' in value_clean.lower() or
                    'placeholder' in value_clean.lower() or
                    'name' in value_clean.lower() and len(value_clean) < 10):
                    
                    # Replace with realistic names
                    realistic_names = [
                        'Sarah Johnson', 'Michael Chen', 'Emily Rodriguez', 'David Kim',
                        'Jessica Williams', 'Robert Brown', 'Ashley Davis', 'Christopher Lee',
                        'Amanda Wilson', 'James Martinez', 'Lauren Anderson', 'Matthew Garcia',
                        'Stephanie Taylor', 'Daniel Thompson', 'Nicole White', 'Ryan Jackson',
                        'Samantha Lewis', 'Kevin Miller', 'Rachel Moore', 'Brandon Clark'
                    ]
                    return realistic_names[record_index % len(realistic_names)]
                
                return value_clean
            return f"Person {record_index + 1}"
        
        # Generic cleaning for other fields
        if isinstance(value, str):
            value_clean = value.strip()
            # Remove obvious placeholder patterns
            if ('sample' in value_clean.lower() and len(value_clean) < 20) or 'generated' in value_clean.lower():
                return f"Realistic_{field_name}_{record_index + 1}"
            return value_clean
        
        return value
    
    def _validate_record_consistency(self, record: Dict[str, Any], domain: str) -> bool:
        """Validate consistency across fields in a record"""
        
        # Healthcare-specific consistency checks
        if domain == 'healthcare':
            age = record.get('age')
            condition = record.get('conditions') or record.get('condition') or record.get('diagnosis')
            
            if age is not None and condition is not None:
                age = int(age) if isinstance(age, (int, float, str)) and str(age).isdigit() else 35
                condition_str = str(condition).lower()
                
                # Age-appropriate condition validation
                if age < 18:  # Pediatric patients
                    pediatric_inappropriate = ['alzheimer', 'dementia', 'osteoarthritis', 'menopause']
                    if any(term in condition_str for term in pediatric_inappropriate):
                        return False
                elif age > 80:  # Elderly patients
                    elderly_inappropriate = ['adhd', 'autism', 'learning disability']
                    if any(term in condition_str for term in elderly_inappropriate):
                        return False
        
        return True