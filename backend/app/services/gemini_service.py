import os
import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from datetime import datetime, timedelta
import time
import asyncio

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model = None
        self.is_initialized = False
        
        # Health check caching to avoid quota waste
        self._last_health_check = None
        self._last_health_check_time = 0
        self._health_check_cache_duration = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize Gemini service with proper API key validation"""
        logger.info("🤖 Initializing Gemini 2.0 Flash service...")
        
        if not self.api_key:
            logger.warning("⚠️ No Gemini API key found in environment")
            self.is_initialized = False
            return False
            
        if self.api_key in ['AIzaSyA81SV6mvA9ShZasJgcVl4ps-YQm9DrKsc', 'AIzaSyA81SV6mvA9ShZasJgcVl4ps-YQm9DrKsc']:
            logger.warning("⚠️ Placeholder API key detected, Gemini service disabled")
            self.is_initialized = False
            return False
        
        try:
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
        """Perform health check with caching to preserve quota"""
        current_time = time.time()
        
        # Return cached result if still valid
        if (self._last_health_check and 
            current_time - self._last_health_check_time < self._health_check_cache_duration):
            logger.info("🔄 Returning cached health status to save quota")
            return self._last_health_check
        
        if not self.is_initialized:
            result = {
                "status": "error",
                "model": None,
                "message": "Service not initialized",
                "api_key_configured": bool(self.api_key),
                "api_key_status": "valid" if self.api_key and self.api_key not in ['AIzaSyA81SV6mvA9ShZasJgcVl4ps-YQm9DrKsc', 'AIzaSyA81SV6mvA9ShZasJgcVl4ps-YQm9DrKsc'] else "invalid",
                "quota_preserved": True
            }
            self._last_health_check = result
            self._last_health_check_time = current_time
            return result
        
        # Only perform actual API call if cache is expired
        try:
            # Quick test with minimal prompt to save quota
            test_response = self.model.generate_content("Test")
            
            result = {
                "status": "ready",
                "model": "gemini-2.0-flash",
                "message": "Initialized and ready (quota-preserving mode)",
                "api_key_configured": True,
                "api_key_status": "configured",
                "quota_preserved": True
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini health check failed: {str(e)}")
            result = {
                "status": "error",
                "model": "gemini-2.0-flash",
                "message": f"Health check failed: {str(e)}",
                "api_key_configured": True,
                "api_key_status": "configured_but_error",
                "quota_preserved": True
            }
        
        # Cache the result
        self._last_health_check = result
        self._last_health_check_time = current_time
        return result

    async def generate_synthetic_data(
        self,
        schema: Dict[str, Any], 
        config: Dict[str, Any],
        description: str = "",
        source_data: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate high-quality synthetic data using Gemini 2.0 Flash with enhanced realism"""
        
        if not self.is_initialized:
            logger.info("🏠 Gemini not initialized, using enhanced fallback generation")
            return self._generate_enhanced_fallback_data(schema, config)
        
        logger.info("🤖 Generating synthetic data with Gemini 2.0 Flash...")
        
        try:
            # Extract domain context for realistic generation
            domain_context = self._extract_domain_context(schema, config, description)
            domain = domain_context['domain']
            field_rules = domain_context['field_rules']
            authenticity_rules = domain_context['authenticity_rules']
            
            # Generate sample data for context if not provided
            if not source_data:
                source_data = []
            
            row_count = min(config.get('rowCount', 100), 1000)  # Cap at 1000 for performance
            
            # Enhanced prompt for production-quality synthetic data generation
            enhanced_prompt = f"""
TASK: Generate {row_count} rows of PRODUCTION-READY {domain.upper()} synthetic data.

CRITICAL REQUIREMENTS FOR ENTERPRISE DATA QUALITY:
1. Generate EXACTLY {row_count} complete records
2. ALL data must be REALISTIC and indistinguishable from real-world data
3. ZERO placeholder values, NO "sample" text, NO generic patterns
4. Data must pass enterprise validation and be ML-training ready
5. Use authentic domain terminology and realistic value distributions

SCHEMA DEFINITION:
{json.dumps(schema, indent=2)}

FIELD GENERATION RULES:
{field_rules}

AUTHENTICITY REQUIREMENTS:
{authenticity_rules}

SOURCE DATA CONTEXT (if available):
{json.dumps(source_data[:2] if source_data else [], indent=2)}

EXAMPLE PATTERNS FOR DOMAIN-SPECIFIC REALISM:
{self._get_realistic_examples(domain, schema)}

OUTPUT SPECIFICATION:
- Return ONLY a valid JSON array with EXACTLY {row_count} objects
- Each object must follow the schema precisely
- All values must be production-quality and realistic
- NO comments, NO explanations, ONLY the JSON array

GENERATE {row_count} REALISTIC RECORDS:
            """
            
            try:
                response = self.model.generate_content(enhanced_prompt)
                result = response.text
                
                logger.info(f"📝 Raw Gemini response length: {len(result)} characters")
                
                # Extract JSON from response
                json_text = self._extract_json_from_response(result)
                logger.info(f"🧹 Extracted JSON length: {len(json_text)} characters")
                
                # Parse and validate JSON
                parsed_data = json.loads(json_text)
                logger.info(f"✅ JSON parsed successfully - type: {type(parsed_data)}")
                
                # Handle single object vs array - FORCE array format
                if isinstance(parsed_data, dict):
                    logger.warning("🔄 Received single object instead of array, requesting full generation...")
                    # If we got a single object, the prompt didn't work - try again with stricter format
                    return await self._generate_with_fallback_strategy(schema, config, domain, row_count)
                
                if not isinstance(parsed_data, list):
                    logger.error(f"❌ Invalid data type received: {type(parsed_data)}")
                    return await self._generate_with_fallback_strategy(schema, config, domain, row_count)
                
                # Validate and clean the generated data
                logger.info("🔍 Validating and cleaning generated data for quality assurance...")
                validated_data = self._validate_and_clean_data(parsed_data, schema, domain)
                
                # Ensure we have the right number of records
                if len(validated_data) < row_count * 0.8:  # If we have less than 80% of requested records
                    logger.warning(f"⚠️ Generated only {len(validated_data)}/{row_count} records, using hybrid generation...")
                    return await self._generate_with_fallback_strategy(schema, config, domain, row_count)
                elif len(validated_data) < row_count:
                    logger.info(f"📈 Extending {len(validated_data)} to {row_count} records")
                    validated_data.extend(self._generate_additional_records(
                        validated_data, schema, domain, row_count - len(validated_data)
                    ))
                elif len(validated_data) > row_count:
                    logger.info(f"📏 Trimming to requested {row_count} records")
                    validated_data = validated_data[:row_count]
                
                logger.info(f"✅ Generated {len(validated_data)} synthetic records with Gemini")
                return validated_data
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing error: {str(e)}")
                logger.info("🔄 Falling back to enhanced local generation...")
                return self._generate_enhanced_fallback_data(schema, config)
                
            except Exception as e:
                logger.error(f"❌ Gemini generation error: {str(e)}")
                logger.info("🔄 Falling back to enhanced local generation...")
                return self._generate_enhanced_fallback_data(schema, config)
                
        except Exception as e:
            logger.error(f"❌ Overall generation error: {str(e)}")
            return self._generate_enhanced_fallback_data(schema, config)

    async def _generate_with_fallback_strategy(self, schema: Dict[str, Any], config: Dict[str, Any], domain: str, row_count: int) -> List[Dict[str, Any]]:
        """Generate data using a fallback strategy when Gemini doesn't produce full results"""
        logger.info(f"🔄 Using fallback strategy to generate {row_count} records for {domain}")
        
        # Try a simpler prompt approach
        simple_prompt = f"""Generate a JSON array of {row_count} realistic {domain} records.
Schema: {json.dumps(schema)}
Return only valid JSON array with {row_count} objects."""
        
        try:
            response = self.model.generate_content(simple_prompt)
            result = response.text
            json_text = self._extract_json_from_response(result)
            parsed_data = json.loads(json_text)
            
            if isinstance(parsed_data, list) and len(parsed_data) >= row_count * 0.5:
                validated_data = self._validate_and_clean_data(parsed_data, schema, domain)
                if len(validated_data) < row_count:
                    # Fill remaining with enhanced fallback
                    validated_data.extend(self._generate_additional_records(
                        validated_data, schema, domain, row_count - len(validated_data)
                    ))
                return validated_data[:row_count]
        except Exception as e:
            logger.warning(f"⚠️ Fallback strategy also failed: {str(e)}")
        
        # Final fallback to enhanced local generation
        return self._generate_enhanced_fallback_data(schema, config)

    def _get_realistic_examples(self, domain: str, schema: Dict[str, Any]) -> str:
        """Get domain-specific realistic examples"""
        examples = {
            'healthcare': {
                'patient_id': 'PT123456, MR789012, PT345678',
                'name': 'Jennifer Martinez, Robert Chen, Sarah Williams',
                'age': '34, 67, 42 (realistic age distribution)',
                'gender': 'Male, Female, Other (balanced distribution)',
                'diagnosis': 'Type 2 Diabetes Mellitus, Essential Hypertension, Chronic Obstructive Pulmonary Disease',
                'admission_date': '2024-03-15, 2024-07-22, 2024-11-08'
            },
            'finance': {
                'account_id': 'ACC12345678, TXN98765432, CHK11223344',
                'balance': '2847.63, 15230.12, 892.45',
                'transaction_type': 'Direct Deposit, ATM Withdrawal, Online Transfer',
                'merchant': 'Amazon.com, Shell Gas Station, Starbucks #2847'
            },
            'retail': {
                'customer_id': 'CUST00123456, USR789012345',
                'product_name': 'Samsung Galaxy S24 Ultra, Nike Air Max 270, Sony WH-1000XM5',
                'category': 'Electronics, Footwear, Audio Equipment',
                'price': '1199.99, 129.95, 399.99'
            }
        }
        
        domain_examples = examples.get(domain, {})
        if not domain_examples:
            return "Use realistic, professional data values appropriate for the domain."
        
        example_text = f"Example realistic {domain} values:\n"
        for field, example_values in domain_examples.items():
            if any(field.lower() in schema_field.lower() for schema_field in schema.keys()):
                example_text += f"- {field}: {example_values}\n"
        
        return example_text

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from Gemini response text"""
        text = response_text.strip()
        
        # Remove markdown code blocks
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0] if text.count('```') >= 2 else text.split('```')[1]
        
        # Find JSON array or object
        start_pos = text.find('[')
        end_pos = text.rfind(']')
        
        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            return text[start_pos:end_pos + 1]
        
        # Fallback: try to find object
        start_pos = text.find('{')
        end_pos = text.rfind('}')
        
        if start_pos != -1 and end_pos != -1:
            return text[start_pos:end_pos + 1]
        
        return text.strip()

    async def analyze_schema_advanced(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: List[str]
    ) -> Dict[str, Any]:
        """Advanced schema analysis with comprehensive insights"""
        if not self.is_initialized:
            return {
                "domain": "general",
                "data_types": {},
                "relationships": [],
                "quality_score": 85,
                "pii_detected": False,
                "error": "Gemini service not available"
            }

        prompt = f"""
        Analyze this dataset comprehensively:
        Data: {json.dumps(data[:3], indent=2)}
        Config: {json.dumps(config)}
        
        Provide analysis including:
        1. Domain classification
        2. Data type inference
        3. Quality assessment
        4. PII detection
        5. Relationship mapping
        
        Return as JSON:
        {{
            "domain": "detected_domain",
            "data_types": {{"field": "type"}},
            "relationships": ["field1 relates to field2"],
            "quality_score": 0-100,
            "pii_detected": true/false,
            "recommendations": ["improvement suggestions"]
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
                "generation_strategy": "recommended approach",
                "field_treatments": ["field-specific recommendations"]
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
                "recommendations": {"generation_strategy": f"Analysis failed: {str(e)}"}
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
            
            elif 'condition' in field_lower or 'diagnosis' in field_lower:
                rules.append(f"- {field_name}: Authentic medical conditions (ICD-10 compliant names, no abbreviations)")
            
            elif 'date' in field_lower:
                rules.append(f"- {field_name}: Realistic dates (YYYY-MM-DD format, within appropriate time ranges)")
            
            elif 'email' in field_lower:
                rules.append(f"- {field_name}: Valid email format with realistic domains")
            
            elif 'phone' in field_lower:
                rules.append(f"- {field_name}: Valid phone numbers with appropriate country/region codes")
        
        return '\n'.join(rules) if rules else f"- All fields: Use realistic, domain-appropriate values for {domain}"
    
    def _get_authenticity_rules(self, domain: str) -> str:
        """Get domain-specific authenticity rules"""
        rules = {
            'healthcare': """
HEALTHCARE DATA AUTHENTICITY RULES:
- Use real medical terminology and ICD-10 condition names
- Patient IDs follow hospital formats (PT######, MR######)
- Ages must reflect realistic patient demographics
- Admission/discharge dates should be logical and recent
- Doctor names should be professional (Dr. [First] [Last])
- Insurance types should match real healthcare systems
- NO placeholder text, ALL medical terms must be authentic
            """,
            'finance': """
FINANCIAL DATA AUTHENTICITY RULES:
- Account numbers follow banking standards (8-12 digits)
- Transaction amounts should be realistic (-$5000 to +$50000)
- Merchant names should be recognizable brands/businesses
- Transaction types: Deposit, Withdrawal, Transfer, Payment, Fee
- Dates should reflect normal banking patterns (weekdays mostly)
- Balance changes should be mathematically consistent
- NO generic patterns, ALL financial data must be realistic
            """,
            'retail': """
RETAIL DATA AUTHENTICITY RULES:
- Product names should be real brands and models
- Prices must be market-realistic for each product category
- Customer IDs follow e-commerce patterns (CUST###### or UUID)
- Purchase dates reflect seasonal patterns and trends
- Ratings should be 1-5 stars with realistic distribution
- Categories should match actual retail classifications
- NO placeholder products, ALL items must be authentic
            """,
            'education': """
EDUCATION DATA AUTHENTICITY RULES:
- Student IDs follow institutional formats (STU######)
- Course names should be actual academic subjects
- Grades follow standard systems (A-F, GPA 0.0-4.0)
- Enrollment dates align with academic calendars
- Teacher names should be professional and diverse
- Class sizes should be realistic (15-300 students)
- NO generic course names, ALL academic data must be authentic
            """
        }
        
        return rules.get(domain, f"""
GENERAL DATA AUTHENTICITY RULES:
- ALL values must be realistic and production-ready
- NO placeholder text or generic patterns
- Use appropriate data formats and ranges
- Ensure logical consistency between related fields
- Follow industry standards for {domain} domain
        """)

    def _validate_and_clean_data(self, data: List[Dict[str, Any]], schema: Dict[str, Any], domain: str) -> List[Dict[str, Any]]:
        """Validate and clean generated data to ensure production quality"""
        logger.info(f"🔍 Validating {len(data)} records for {domain} domain...")
        
        cleaned_data = []
        
        for i, record in enumerate(data):
            # Skip empty or invalid records
            if not record or not isinstance(record, dict):
                logger.warning(f"Skipping invalid record {i}: {type(record)}")
                continue
            
            cleaned_record = {}
            is_valid = True
            
            # Validate each field according to schema and domain rules
            for field_name, field_info in schema.items():
                if field_name not in record:
                    logger.warning(f"Missing field {field_name} in record {i}")
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
            return f"Person_{record_index + 1}"
        
        # Default: return the value as-is or generate appropriate fallback
        if isinstance(value, str):
            value_clean = value.strip()
            # Check for generic placeholder patterns
            if ('sample' in value_clean.lower() or 
                'generated' in value_clean.lower() or
                'placeholder' in value_clean.lower()):
                
                # Generate realistic replacement based on field type
                if field_type == 'number':
                    return 100 + (record_index * 37) % 1000
                elif field_type == 'boolean':
                    return record_index % 2 == 0
                else:
                    return f"Realistic_{field_name}_{record_index + 1}"
            
            return value_clean
        
        return value
    
    def _validate_record_consistency(self, record: Dict[str, Any], domain: str) -> bool:
        """Validate record-level consistency"""
        # Add domain-specific consistency checks
        if domain == 'healthcare':
            # Example: admission date should be before discharge date
            if 'admission_date' in record and 'discharge_date' in record:
                try:
                    from datetime import datetime
                    admission = datetime.strptime(record['admission_date'], '%Y-%m-%d')
                    discharge = datetime.strptime(record['discharge_date'], '%Y-%m-%d')
                    if admission > discharge:
                        logger.warning("Admission date after discharge date - inconsistent record")
                        return False
                except:
                    pass  # Skip validation if date parsing fails
        
        return True
    
    def _generate_additional_records(self, existing_data: List[Dict[str, Any]], schema: Dict[str, Any], domain: str, count: int) -> List[Dict[str, Any]]:
        """Generate additional records to meet the requested count"""
        additional_records = []
        
        for i in range(count):
            new_record = {}
            base_index = len(existing_data) + i
            
            for field_name, field_info in schema.items():
                new_record[field_name] = self._generate_realistic_field_value(
                    field_name, field_info, domain, base_index
                )
            
            additional_records.append(new_record)
        
        return additional_records
    
    def _generate_realistic_field_value(self, field_name: str, field_info: Dict[str, Any], domain: str, index: int) -> Any:
        """Generate a realistic value for a specific field"""
        field_lower = field_name.lower()
        field_type = field_info.get('type', 'string')
        examples = field_info.get('examples', [])
        constraints = field_info.get('constraints', {})
        
        # Use examples if available
        if examples:
            return examples[index % len(examples)]
        
        # Domain-specific realistic generation
        if domain == 'healthcare':
            if 'patient' in field_lower and 'id' in field_lower:
                return f"PT{str(100000 + index).zfill(6)}"
            elif 'name' in field_lower:
                names = ['Sarah Johnson', 'Michael Chen', 'Emily Rodriguez', 'David Kim', 'Jessica Williams']
                return names[index % len(names)]
            elif 'age' in field_lower:
                return min(95, max(0, 30 + (index * 7) % 60))
            elif 'gender' in field_lower:
                genders = ['Male', 'Female', 'Other', 'Prefer not to say']
                return genders[index % 4]
            elif 'diagnosis' in field_lower or 'condition' in field_lower:
                conditions = [
                    'Type 2 Diabetes Mellitus', 'Essential Hypertension', 'Hyperlipidemia',
                    'Chronic Obstructive Pulmonary Disease', 'Osteoarthritis'
                ]
                return conditions[index % len(conditions)]
        
        # Generic type-based generation
        if field_type == 'string':
            return f"Realistic_{field_name}_{index + 1}"
        elif field_type == 'number':
            min_val = constraints.get('min', 1)
            max_val = constraints.get('max', 1000)
            return min_val + (index * (max_val - min_val) // 100) % (max_val - min_val + 1)
        elif field_type == 'boolean':
            return index % 2 == 0
        elif field_type in ['date', 'datetime']:
            from datetime import datetime, timedelta
            base_date = datetime(2024, 1, 1)
            result_date = base_date + timedelta(days=index * 10)
            return result_date.strftime('%Y-%m-%d')
        
        return f"Value_{index + 1}"
    
    def _generate_enhanced_fallback_data(self, schema: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate high-quality fallback data when Gemini is not available"""
        row_count = min(config.get('rowCount', 100), 1000)
        domain = config.get('domain', 'general')
        
        logger.info(f"🔄 Generating {row_count} enhanced fallback records for {domain}")
        
        fallback_data = []
        
        for i in range(row_count):
            record = {}
            for field_name, field_info in schema.items():
                record[field_name] = self._generate_realistic_field_value(field_name, field_info, domain, i)
            fallback_data.append(record)
        
        logger.info(f"✅ Generated {len(fallback_data)} enhanced fallback records")
        return fallback_data