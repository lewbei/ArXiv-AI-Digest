"""
Unified Model Interface for Academic Paper Summarization

This module provides a unified interface for interacting with multiple AI model providers,
including cloud APIs, enterprise solutions, and local models.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime


class ModelProvider(Enum):
    """Supported model providers"""
    # Cloud APIs
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    PERPLEXITY = "perplexity"
    XAI = "xai"
    MISTRAL = "mistral"
    
    # Enterprise
    AZURE_OPENAI = "azure"
    AWS_BEDROCK = "bedrock"
    VERTEX_AI = "vertex"
    
    # Local/Self-hosted
    OLLAMA = "ollama"
    
    # Aggregators
    OPENROUTER = "openrouter"
    
    # CLI Integrations (no task-master dependency)
    CLAUDE_CODE = "claude_code"
    GEMINI_CLI = "gemini_cli"


class ModelCapability(Enum):
    """Model capabilities for intelligent routing"""
    TEXT_GENERATION = "text_generation"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"
    HIGH_QUALITY = "high_quality"
    COST_EFFECTIVE = "cost_effective"
    RESEARCH_OPTIMIZED = "research_optimized"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_id: str
    display_name: str
    max_tokens: int = 4000
    context_window: int = 32000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[ModelCapability] = field(default_factory=list)
    swe_score: Optional[float] = None
    is_free: bool = False
    supports_system_prompt: bool = True
    supports_json_output: bool = False
    api_endpoint: Optional[str] = None
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request"""
        if self.is_free:
            return 0.0
        return (input_tokens * self.cost_per_1k_input / 1000 + 
                output_tokens * self.cost_per_1k_output / 1000)


@dataclass
class ModelRequest:
    """Request to a model"""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    json_output: bool = False
    model_specific_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from a model"""
    content: str
    model_id: str
    provider: ModelProvider
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    finish_reason: str
    response_time: float
    estimated_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelError:
    """Error information from model calls"""
    provider: ModelProvider
    model_id: str
    error_type: str  # rate_limit, auth, network, model_error, etc.
    message: str
    retry_after: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ModelClient(ABC):
    """Abstract base class for model clients"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.provider.value}")
    
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate text using the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is currently available"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        pass


class ModelRegistry:
    """Registry of available models with their configurations"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default model configurations"""
        
        # Anthropic Models
        self.register_model(ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            max_tokens=8192,
            context_window=200000,
            cost_per_1k_input=3.0,
            cost_per_1k_output=15.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.LONG_CONTEXT, ModelCapability.RESEARCH_OPTIMIZED],
            swe_score=72.7,
            supports_json_output=True
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            max_tokens=8192,
            context_window=200000,
            cost_per_1k_input=15.0,
            cost_per_1k_output=75.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.LONG_CONTEXT, ModelCapability.RESEARCH_OPTIMIZED],
            swe_score=72.5,
            supports_json_output=True
        ))
        
        # OpenAI Models
        self.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
            display_name="GPT-4o",
            max_tokens=4096,
            context_window=128000,
            cost_per_1k_input=2.5,
            cost_per_1k_output=10.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.FAST_INFERENCE],
            swe_score=33.2,
            supports_json_output=True
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="o1",
            display_name="OpenAI o1",
            max_tokens=32768,
            context_window=200000,
            cost_per_1k_input=15.0,
            cost_per_1k_output=60.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.RESEARCH_OPTIMIZED],
            swe_score=48.9
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="o3",
            display_name="OpenAI o3",
            max_tokens=32768,
            context_window=200000,
            cost_per_1k_input=2.0,
            cost_per_1k_output=8.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.RESEARCH_OPTIMIZED],
            swe_score=50.0
        ))
        
        # Google Models
        self.register_model(ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            max_tokens=8192,
            context_window=2000000,
            cost_per_1k_input=0.0,  # Free tier
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.LONG_CONTEXT, ModelCapability.COST_EFFECTIVE],
            swe_score=72.0,
            is_free=True,
            supports_json_output=True
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_id="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            max_tokens=8192,
            context_window=1000000,
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            capabilities=[ModelCapability.FAST_INFERENCE, ModelCapability.COST_EFFECTIVE],
            swe_score=51.8,
            supports_json_output=True
        ))
        
        # Perplexity Models
        self.register_model(ModelConfig(
            provider=ModelProvider.PERPLEXITY,
            model_id="sonar-pro",
            display_name="Perplexity Sonar Pro",
            max_tokens=4096,
            context_window=127000,
            cost_per_1k_input=3.0,
            cost_per_1k_output=15.0,
            capabilities=[ModelCapability.RESEARCH_OPTIMIZED],
            supports_json_output=True
        ))
        
        # Local/Ollama Models
        self.register_model(ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_id="llama3.3:latest",
            display_name="Llama 3.3 70B",
            max_tokens=4096,
            context_window=128000,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.COST_EFFECTIVE, ModelCapability.HIGH_QUALITY],
            is_free=True,
            api_endpoint="http://localhost:11434"
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_id="qwen3:latest",
            display_name="Qwen 3",
            max_tokens=4096,
            context_window=32768,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.COST_EFFECTIVE, ModelCapability.FAST_INFERENCE],
            is_free=True,
            api_endpoint="http://localhost:11434"
        ))
        
            # Claude Code and Gemini CLI Integration Models
        self.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE_CODE,
            model_id="sonnet",
            display_name="Claude Code Sonnet",
            max_tokens=8192,
            context_window=200000,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.RESEARCH_OPTIMIZED],
            swe_score=72.7,
            is_free=True
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE_CODE,
            model_id="opus",
            display_name="Claude Code Opus",
            max_tokens=8192,
            context_window=200000,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.RESEARCH_OPTIMIZED],
            swe_score=72.5,
            is_free=True
        ))
        
        self.register_model(ModelConfig(
            provider=ModelProvider.GEMINI_CLI,
            model_id="gemini-2.5-pro",
            display_name="Gemini CLI 2.5 Pro",
            max_tokens=8192,
            context_window=1000000,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.LONG_CONTEXT, ModelCapability.COST_EFFECTIVE],
            swe_score=72.0,
            is_free=True
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a new model configuration"""
        key = f"{config.provider.value}:{config.model_id}"
        self.models[key] = config
    
    def get_model(self, provider: ModelProvider, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration"""
        key = f"{provider.value}:{model_id}"
        return self.models.get(key)
    
    def list_models(self, provider: Optional[ModelProvider] = None, 
                   capability: Optional[ModelCapability] = None,
                   max_cost: Optional[float] = None) -> List[ModelConfig]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        if max_cost is not None:
            models = [m for m in models if m.is_free or m.cost_per_1k_output <= max_cost]
        
        return models
    
    def get_best_model(self, 
                      capability: ModelCapability,
                      max_cost: Optional[float] = None,
                      prefer_free: bool = True) -> Optional[ModelConfig]:
        """Get the best model for a specific capability"""
        candidates = self.list_models(capability=capability, max_cost=max_cost)
        
        if not candidates:
            return None
        
        # Prefer free models if requested
        if prefer_free:
            free_models = [m for m in candidates if m.is_free]
            if free_models:
                candidates = free_models
        
        # Sort by SWE score (if available) or other quality metrics
        candidates.sort(key=lambda m: (
            m.swe_score or 0,
            1 if m.is_free else 0,
            -m.cost_per_1k_output
        ), reverse=True)
        
        return candidates[0]


class IntelligentModelSelector:
    """Intelligent model selection based on request characteristics"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def select_model(self, 
                    request: ModelRequest,
                    paper_analysis: Optional[Dict] = None,
                    user_preferences: Optional[Dict] = None) -> ModelConfig:
        """Select optimal model based on request and context"""
        
        # Determine required capabilities
        required_capabilities = self._analyze_requirements(request, paper_analysis)
        
        # Get user constraints
        max_cost = user_preferences.get('max_cost') if user_preferences else None
        prefer_free = user_preferences.get('prefer_free', True) if user_preferences else True
        
        # Find best model for primary capability
        primary_capability = required_capabilities[0] if required_capabilities else ModelCapability.HIGH_QUALITY
        
        selected_model = self.registry.get_best_model(
            capability=primary_capability,
            max_cost=max_cost,
            prefer_free=prefer_free
        )
        
        # Fallback to any available model
        if not selected_model:
            all_models = self.registry.list_models()
            if all_models:
                selected_model = all_models[0]
        
        if not selected_model:
            raise ValueError("No available models found")
        
        return selected_model
    
    def _analyze_requirements(self, 
                            request: ModelRequest,
                            paper_analysis: Optional[Dict] = None) -> List[ModelCapability]:
        """Analyze request to determine required capabilities"""
        capabilities = []
        
        # Check for long context requirement
        estimated_tokens = len(request.prompt.split()) * 1.3  # Rough token estimation
        if estimated_tokens > 16000:
            capabilities.append(ModelCapability.LONG_CONTEXT)
        
        # Check for structured output requirement
        if request.json_output:
            capabilities.append(ModelCapability.HIGH_QUALITY)
        
        # Check paper complexity
        if paper_analysis:
            complexity = paper_analysis.get('technical_complexity', 0)
            if complexity > 0.5:
                capabilities.append(ModelCapability.RESEARCH_OPTIMIZED)
            elif complexity < 0.2:
                capabilities.append(ModelCapability.FAST_INFERENCE)
        
        # Default to high quality if no specific requirements
        if not capabilities:
            capabilities.append(ModelCapability.HIGH_QUALITY)
        
        return capabilities


class ModelOrchestrator:
    """Orchestrates multiple model clients with failover and load balancing"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.selector = IntelligentModelSelector(self.registry)
        self.clients: Dict[str, ModelClient] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_client(self, client: ModelClient):
        """Register a model client"""
        key = f"{client.config.provider.value}:{client.config.model_id}"
        self.clients[key] = client
    
    async def generate(self, 
                      request: ModelRequest,
                      paper_analysis: Optional[Dict] = None,
                      user_preferences: Optional[Dict] = None,
                      fallback_attempts: int = 3) -> ModelResponse:
        """Generate text with automatic model selection and failover"""
        
        # Select primary model
        selected_model = self.selector.select_model(request, paper_analysis, user_preferences)
        
        # Attempt generation with fallback
        attempts = 0
        last_error = None
        
        # Try primary model first
        models_to_try = [selected_model]
        
        # Add fallback models
        if fallback_attempts > 1:
            fallback_models = self.registry.list_models()
            # Remove primary model from fallbacks
            fallback_models = [m for m in fallback_models if m != selected_model]
            # Sort by quality/availability
            fallback_models.sort(key=lambda m: (m.is_free, m.swe_score or 0), reverse=True)
            models_to_try.extend(fallback_models[:fallback_attempts-1])
        
        for model_config in models_to_try:
            attempts += 1
            client_key = f"{model_config.provider.value}:{model_config.model_id}"
            
            if client_key not in self.clients:
                self.logger.warning(f"No client available for {client_key}")
                continue
            
            client = self.clients[client_key]
            
            if not client.is_available():
                self.logger.warning(f"Client {client_key} is not available")
                continue
            
            try:
                self.logger.info(f"Attempting generation with {client_key} (attempt {attempts})")
                response = await client.generate(request)
                
                # Add metadata about selection
                response.metadata['selection_reason'] = f"Attempt {attempts}"
                response.metadata['total_attempts'] = attempts
                
                return response
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Generation failed with {client_key}: {str(e)}")
                
                # If this is a rate limit error, wait before next attempt
                if 'rate limit' in str(e).lower():
                    await asyncio.sleep(min(attempts * 2, 30))  # Exponential backoff, max 30s
        
        # All attempts failed
        error_msg = f"All {attempts} generation attempts failed"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        
        raise RuntimeError(error_msg)
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of models with available clients"""
        available = []
        for config in self.registry.list_models():
            client_key = f"{config.provider.value}:{config.model_id}"
            if client_key in self.clients and self.clients[client_key].is_available():
                available.append(config)
        return available
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        total_models = len(self.registry.list_models())
        available_models = len(self.get_available_models())
        
        provider_status = {}
        for provider in ModelProvider:
            models = self.registry.list_models(provider=provider)
            available = sum(1 for m in models 
                          if f"{m.provider.value}:{m.model_id}" in self.clients 
                          and self.clients[f"{m.provider.value}:{m.model_id}"].is_available())
            provider_status[provider.value] = {
                'total': len(models),
                'available': available
            }
        
        return {
            'total_models': total_models,
            'available_models': available_models,
            'provider_status': provider_status,
            'timestamp': datetime.now().isoformat()
        }