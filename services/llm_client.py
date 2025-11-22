"""
LLM Client Wrapper for KB Generation Pipeline

"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
import time


from config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers
    
    Features:
    - Provider abstraction (OpenAI, Anthropic, Google, Ollama)
    - JSON response parsing with validation
    - Automatic retries on failure
    - Token counting and cost tracking
    - Error handling and logging
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client from configuration
        
        Args:
            config: LLMConfig object with all settings
        """
        self.config = config
        
        # Get all settings from config
        self.provider = config.provider
        self.model = config.get_model()
        self.api_key = config.get_api_key()
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.base_url = config.ollama_base_url if config.provider == LLMProvider.OLLAMA else None
        
        # Validate we have API key (if needed)
        if self.provider != LLMProvider.OLLAMA and not self.api_key:
            raise ValueError(
                f"API key required for {self.provider.value}. "
                f"Set it in .env file or pass via config."
            )
        
        # Initialize provider-specific client
        self.client = None
        self._init_client()
        
        # Track usage
        self.total_tokens = 0
        self.total_calls = 0
    
        logger.info(f"Initialized LLMClient: provider={self.provider.value}, model={self.model}")


    def _init_client(self):
        """Initialize provider-specific client"""
        try:
            if self.provider == LLMProvider.OPENAI:
                self._init_openai()
            elif self.provider == LLMProvider.ANTHROPIC:
                self._init_anthropic()
            elif self.provider == LLMProvider.GOOGLE:
                self._init_google()
            elif self.provider == LLMProvider.OLLAMA:
                self._init_ollama()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except ImportError as e:
            raise ImportError(
                f"Required package not installed for {self.provider}. "
                f"Install with: pip install {self._get_required_package()}"
            ) from e
    
    def _get_required_package(self) -> str:
        """Get required package name for provider"""
        return self.config.required_packages.get(self.provider, "unknown")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        logger.debug("OpenAI client initialized")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        from anthropic import Anthropic
        
        self.client = Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        logger.debug("Anthropic client initialized")
    
    def _init_google(self):
        """Initialize Google GenAI client (new SDK)"""
        from google import genai
        
        # Create client with API key
        # Client automatically uses GOOGLE_API_KEY or GEMINI_API_KEY from env
        self.client = genai.Client(api_key=self.api_key)
        
        logger.debug("Google GenAI client initialized (new SDK)")
    
    def _init_ollama(self):
        """Initialize Ollama client (local Llama)"""
        import ollama
        
        # Ollama client doesn't need explicit initialization
        # It connects to local Ollama server (default: http://localhost:11434)
        self.client = ollama
        
        # Set base URL if provided
        if self.base_url:
            # Ollama uses environment variable for custom URL
            os.environ['OLLAMA_HOST'] = self.base_url
        
        logger.debug(f"Ollama client initialized (base_url: {self.base_url or 'default'})")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions for LLM)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop_sequences: List of sequences to stop generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If generation fails after all retries
        """
        # Use provided values or fall back to defaults
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Log the request
        logger.debug(f"Generating response with {self.provider}")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Call provider-specific method
        try:
            if self.provider == LLMProvider.OPENAI:
                response = self._generate_openai(
                    prompt, system_prompt, temperature, max_tokens, stop_sequences, response_format, **kwargs
                )
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self._generate_anthropic(
                    prompt, system_prompt, temperature, max_tokens, stop_sequences, response_format, **kwargs
                )
            elif self.provider == LLMProvider.GOOGLE:
                response = self._generate_google(
                    prompt, system_prompt, temperature, max_tokens, stop_sequences, response_format, **kwargs
                )
            elif self.provider == LLMProvider.OLLAMA:
                response = self._generate_ollama(
                    prompt, system_prompt, temperature, max_tokens, stop_sequences, response_format, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Track usage
            self.total_calls += 1
            logger.debug(f"Response length: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        response_format: Optional[str] = None,  # ADD THIS
        **kwargs
    ) -> str:
        """Generate with OpenAI"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Build request parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add JSON mode if requested
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}
        
        if stop_sequences:
            params["stop"] = stop_sequences
        
        # Add any extra kwargs
        params.update(kwargs)
        
        # Call API
        response = self.client.chat.completions.create(**params)
        
        # Track tokens
        if hasattr(response, 'usage'):
            self.total_tokens += response.usage.total_tokens
            logger.debug(f"Tokens used: {response.usage.total_tokens}")
    
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        response_format: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate with Anthropic Claude"""
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        if response_format == "json":
            prompt += "\n\nIMPORTANT: Respond with valid JSON only."
        
        if system_prompt:
            params["system"] = system_prompt
        
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        
        # Add any extra kwargs
        params.update(kwargs)
        
        # Call API
        response = self.client.messages.create(**params)
        
        # Track tokens
        if hasattr(response, 'usage'):
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            logger.debug(f"Tokens used: {response.usage.input_tokens + response.usage.output_tokens}")
        
        return response.content[0].text
    
    def _generate_google(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        response_format: Optional[str] = None,  # ADD THIS
        **kwargs
    ) -> str:
        """Generate with Google GenAI (new SDK)"""
        from google.genai import types
        
        # Add JSON instruction to prompt if needed
        if response_format == "json":
            prompt += "\n\nIMPORTANT: Respond with valid JSON only. Do not include markdown formatting or code blocks."
        
        # Build config parameters
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Add system instruction if provided
        if system_prompt:
            config_params["system_instruction"] = system_prompt
        
        # Add stop sequences if provided
        if stop_sequences:
            config_params["stop_sequences"] = stop_sequences
        
        # Add any extra kwargs (but filter out response_format)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'response_format'}
        config_params.update(filtered_kwargs)
        
        # Create config object
        config = types.GenerateContentConfig(**config_params)
        
        # Call API with new SDK structure
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        # Track tokens if available
        if hasattr(response, 'usage_metadata'):
            if hasattr(response.usage_metadata, 'total_token_count'):
                total = response.usage_metadata.total_token_count
                self.total_tokens += total
                logger.debug(f"Tokens used: {total}")
        
        return response.text
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        response_format: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate with Ollama (local Llama)"""

        if response_format == "json":
            prompt += "\n\nIMPORTANT: Respond with valid JSON only."

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Build options
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
        }
        
        if stop_sequences:
            options["stop"] = stop_sequences
        
        options.update(kwargs)
        
        # Call Ollama API
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options
        )
        
        # Track tokens (approximate)
        if 'eval_count' in response:
            self.total_tokens += response.get('eval_count', 0)
            logger.debug(f"Tokens used: {response.get('eval_count', 0)}")
        
        return response['message']['content']
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Generate JSON response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            schema: JSON schema for structured output
            max_retries: Maximum retry attempts
            
        Returns:
            Parsed JSON response
        """
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_format="json"
                )
                
                # Parse JSON
                try:
                    result = json.loads(response)
                    if schema:
                        self._validate_json_schema(result, schema)
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise ValueError(f"Invalid JSON response after {max_retries} attempts")
            
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a 503 or rate limit error
                if '503' in error_msg or 'overloaded' in error_msg.lower() or 'rate limit' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 3  # Longer wait for 503
                        logger.warning(f"Service overloaded, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Service still overloaded after {max_retries} attempts")
                        raise
                else:
                    # Not a retry-able error
                    raise
        
        raise Exception("Max retries exceeded")


    def _validate_json_schema(self, data: Dict, schema: Dict):
        """
        Validate JSON against schema
        
        Basic validation checking required fields
        For full JSON schema validation, use jsonschema library
        """
        required = schema.get('required', [])
        
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field in JSON response: {field}")
        
        logger.debug(f"JSON schema validation passed ({len(required)} required fields)")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        This is a rough estimation. For accurate counts:
        - OpenAI: Use tiktoken library
        - Anthropic: Use anthropic.count_tokens()
        - Others: Use provider-specific methods
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token for English text
        # This works reasonably well for most models
        estimated = len(text) // 4
        
        logger.debug(f"Estimated tokens: {estimated} (chars: {len(text)})")
        return estimated
    
    def get_token_limit(self) -> int:
        """Get context window token limit for current model"""
        return self.config.token_limits.get(self.model, 4096)
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics
        
        Returns:
            Dictionary with total_calls and total_tokens
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_call": self.total_tokens // self.total_calls if self.total_calls > 0 else 0
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.total_tokens = 0
        self.total_calls = 0
        logger.debug("Usage statistics reset")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current client configuration"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider.value}, model={self.model})"


# Convenience functions for quick initialization

def create_llm_client(config: LLMConfig) -> LLMClient:
    """
    Convenience function to create LLMClient
    
    Args:
        config: LLMConfig object
        
    Example:
        from config import LLMConfig, LLMProvider
        
        config = LLMConfig()
        config.provider = LLMProvider.GOOGLE
        llm = create_llm_client(config)
    """
    return LLMClient(config=config)



# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from config import LLMConfig, LLMProvider, load_env_file
    
    # Load environment variables
    load_env_file()
    
    print("LLM Client Wrapper - Example Usage\n")
    
    # Example 1: Google Gemini (Default)
    print("=" * 60)
    print("Example 1: Google Gemini")
    print("=" * 60)
    try:
        config = LLMConfig()
        config.provider = LLMProvider.GOOGLE
        llm = LLMClient(config)
        
        response = llm.generate(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful geography assistant. Answer concisely."
        )
        print(f"Response: {response}\n")
        
        # JSON generation
        json_response = llm.generate_json(
            prompt="List three colors with their RGB values as a JSON object."
        )
        print(f"JSON Response: {json.dumps(json_response, indent=2)}\n")
        print(f"Usage stats: {llm.get_usage_stats()}\n")
        
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: OpenAI
    print("=" * 60)
    print("Example 2: OpenAI GPT-4")
    print("=" * 60)
    try:
        config = LLMConfig()
        config.provider = LLMProvider.OPENAI
        config.model = "gpt-4o-mini"
        llm = LLMClient(config)
        
        response = llm.generate(
            prompt="Explain machine learning in one sentence.",
            system_prompt="You are a technical educator."
        )
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)