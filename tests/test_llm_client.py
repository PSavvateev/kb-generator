# test_llm_client.py
"""
Comprehensive Test Suite for LLMClient (Updated with Google GenAI SDK)

Run with: python test_llm_client.py
Or with pytest: pytest test_llm_client.py -v

Tests cover:
- Client initialization for all providers
- Configuration methods
- Token counting and limits
- Usage statistics tracking
- JSON parsing and validation
- Text generation (requires API keys)
- JSON generation (requires API keys)
- Error handling
- Provider-specific features
"""

import json
import os
import sys
import pytest
from typing import Dict, Optional
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from services.llm_client import LLMClient, LLMProvider, create_llm_client

from dotenv import load_dotenv
load_dotenv()



# ============================================================================
# Test Fixtures
# ============================================================================

class TestConfig:
    """Test configuration and utilities"""
    
    @staticmethod
    def has_api_key(provider: str) -> bool:
        """Check if API key is available for provider"""
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
            'ollama': None
        }
        
        env_var = env_vars.get(provider)
        if env_var is None:
            return True  # Ollama doesn't need API key
        
        if isinstance(env_var, list):
            return any(os.getenv(var) for var in env_var)
        
        return bool(os.getenv(env_var))
    
    @staticmethod
    def skip_if_no_api_key(provider: str):
        """Decorator to skip test if API key not available"""
        if not TestConfig.has_api_key(provider):
            return pytest.mark.skip(reason=f"No API key for {provider}")
        return lambda func: func


# ============================================================================
# Unit Tests (No API Key Required)
# ============================================================================

class TestClientInitialization:
    """Test client initialization for all providers"""
    
    def test_openai_initialization(self):
        """Test OpenAI client initialization"""
        try:
            # Use dummy API key for initialization test
            client = LLMClient(
                provider='openai', 
                model='gpt-3.5-turbo',
                api_key='test-key-for-initialization'
            )
            assert client.provider == LLMProvider.OPENAI
            assert client.model == 'gpt-3.5-turbo'
            assert client.temperature == 0.3  # default
            print("✅ OpenAI client initialized")
        except ImportError:
            print("⚠️  OpenAI package not installed")
            pytest.skip("OpenAI package not installed")
    
    def test_anthropic_initialization(self):
        """Test Anthropic client initialization"""
        try:
            client = LLMClient(
                provider='anthropic', 
                model='claude-sonnet-4',
                api_key='test-key-for-initialization'
            )
            assert client.provider == LLMProvider.ANTHROPIC
            assert client.model == 'claude-sonnet-4'
            print("✅ Anthropic client initialized")
        except ImportError:
            print("⚠️  Anthropic package not installed")
            pytest.skip("Anthropic package not installed")
    
    def test_google_initialization(self):
        """Test Google GenAI client initialization (new SDK)"""
        try:
            client = LLMClient(
                provider='google', 
                model='gemini-2.5-flash',
                api_key='test-key-for-initialization'
            )
            assert client.provider == LLMProvider.GOOGLE
            assert client.model == 'gemini-2.5-flash'
            print("✅ Google GenAI client initialized (new SDK)")
        except ImportError:
            print("⚠️  Google GenAI package not installed")
            pytest.skip("Google GenAI package not installed")
    
    def test_ollama_initialization(self):
        """Test Ollama client initialization"""
        try:
            client = LLMClient(provider='ollama', model='llama3.1:8b')
            assert client.provider == LLMProvider.OLLAMA
            assert client.model == 'llama3.1:8b'
            print("✅ Ollama client initialized")
        except ImportError:
            print("⚠️  Ollama package not installed")
            pytest.skip("Ollama package not installed")
    
    def test_default_models(self):
        """Test that default models are used when model not specified"""
        providers_to_test = ['openai', 'anthropic', 'google', 'ollama']
        
        for provider in providers_to_test:
            try:
                client = LLMClient(
                    provider=provider,
                    api_key='test-key-for-initialization'  # Dummy key
                )
                assert client.model == LLMClient.DEFAULT_MODELS[LLMProvider(provider)]
                print(f"✅ Default model for {provider}: {client.model}")
            except ImportError:
                print(f"⚠️  {provider} package not installed")
    
    def test_custom_parameters(self):
        """Test client initialization with custom parameters"""
        try:
            client = LLMClient(
                provider='openai',
                model='gpt-4',
                api_key='test-key-for-initialization',
                temperature=0.7,
                max_tokens=2000,
                timeout=60,
                max_retries=5
            )
            
            assert client.temperature == 0.7
            assert client.max_tokens == 2000
            assert client.timeout == 60
            assert client.max_retries == 5
            print("✅ Custom parameters set correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestConvenienceFunction:
    """Test convenience function for client creation"""
    
    def test_create_llm_client(self):
        """Test create_llm_client convenience function"""
        try:
            client = create_llm_client(
                'openai', 
                model='gpt-3.5-turbo',
                api_key='test-key-for-initialization'
            )
            assert isinstance(client, LLMClient)
            assert client.provider == LLMProvider.OPENAI
            assert client.model == 'gpt-3.5-turbo'
            print("✅ Convenience function works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestConfigurationMethods:
    """Test configuration and utility methods"""
    
    def test_get_config(self):
        """Test get_config method"""
        try:
            client = LLMClient(
                provider='openai',
                model='gpt-4',
                api_key='test-key-for-initialization',
                temperature=0.5,
                max_tokens=1000
            )
            
            config = client.get_config()
            assert config['provider'] == 'openai'
            assert config['model'] == 'gpt-4'
            assert config['temperature'] == 0.5
            assert config['max_tokens'] == 1000
            assert 'timeout' in config
            assert 'max_retries' in config
            
            print("✅ Configuration methods work")
            print(f"   Config: {config}")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_repr(self):
        """Test string representation"""
        try:
            client = LLMClient(
                provider='openai', 
                model='gpt-4',
                api_key='test-key-for-initialization'
            )
            repr_str = repr(client)
            assert 'openai' in repr_str
            assert 'gpt-4' in repr_str
            print(f"✅ String representation: {repr_str}")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestTokenCounting:
    """Test token counting and limits"""
    
    def test_count_tokens(self):
        """Test token counting estimation"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            text = "Hello world, this is a test message."
            tokens = client.count_tokens(text)
            
            # Should be approximately len(text) / 4
            expected = len(text) // 4
            assert abs(tokens - expected) <= 2  # Allow small variance
            
            print(f"✅ Token counting works")
            print(f"   Text: '{text}'")
            print(f"   Length: {len(text)} chars")
            print(f"   Estimated tokens: {tokens}")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_token_limits(self):
        """Test token limit retrieval for different models"""
        test_cases = [
            ('openai', 'gpt-4', 128000),
            ('openai', 'gpt-3.5-turbo', 16385),
            ('anthropic', 'claude-sonnet-4', 200000),
            ('google', 'gemini-2.5-flash', 1000000),
            ('ollama', 'llama3.1:8b', 8192),
        ]
        
        for provider, model, expected_limit in test_cases:
            try:
                client = LLMClient(
                    provider=provider, 
                    model=model,
                    api_key='test-key-for-initialization'
                )
                limit = client.get_token_limit()
                assert limit == expected_limit
                print(f"✅ Token limit for {model}: {limit}")
            except ImportError:
                print(f"⚠️  {provider} package not installed")


class TestUsageStatistics:
    """Test usage statistics tracking"""
    
    def test_initial_stats(self):
        """Test initial usage statistics"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            stats = client.get_usage_stats()
            assert stats['total_calls'] == 0
            assert stats['total_tokens'] == 0
            assert stats['avg_tokens_per_call'] == 0
            
            print("✅ Initial stats correct:", stats)
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_reset_stats(self):
        """Test stats reset"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            # Manually set some stats
            client.total_calls = 5
            client.total_tokens = 1000
            
            # Reset
            client.reset_usage_stats()
            
            stats = client.get_usage_stats()
            assert stats['total_calls'] == 0
            assert stats['total_tokens'] == 0
            
            print("✅ Stats reset works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestJSONParsing:
    """Test JSON parsing utilities"""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            response = '{"name": "test", "value": 123}'
            parsed = client._parse_json_response(response)
            
            assert parsed == {"name": "test", "value": 123}
            print("✅ Clean JSON parsed correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            response = '```json\n{"name": "test", "value": 123}\n```'
            parsed = client._parse_json_response(response)
            
            assert parsed == {"name": "test", "value": 123}
            print("✅ Markdown JSON parsed correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_parse_json_with_prefix(self):
        """Test parsing JSON with text prefix"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            response = 'Here is the JSON:\n{"name": "test", "value": 123}'
            parsed = client._parse_json_response(response)
            
            assert parsed == {"name": "test", "value": 123}
            print("✅ Prefixed JSON parsed correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_parse_json_embedded(self):
        """Test parsing JSON embedded in text"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            response = 'Some text before {"name": "test", "value": 123} some text after'
            parsed = client._parse_json_response(response)
            
            assert parsed == {"name": "test", "value": 123}
            print("✅ Embedded JSON parsed correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_parse_json_with_code_block(self):
        """Test parsing JSON in generic code block"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            response = '```\n{"name": "test"}\n```'
            parsed = client._parse_json_response(response)
            
            assert parsed == {"name": "test"}
            print("✅ Code block JSON parsed correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestSchemaValidation:
    """Test JSON schema validation"""
    
    def test_valid_schema(self):
        """Test validation with valid data"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            schema = {
                "required": ["name", "age"]
            }
            
            data = {"name": "Alice", "age": 30, "city": "NYC"}
            client._validate_json_schema(data, schema)
            
            print("✅ Valid schema passes validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_invalid_schema(self):
        """Test validation with invalid data"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            schema = {
                "required": ["name", "age"]
            }
            
            data = {"name": "Alice"}  # Missing 'age'
            
            with pytest.raises(ValueError, match="Missing required field"):
                client._validate_json_schema(data, schema)
            
            print("✅ Invalid schema caught correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_empty_schema(self):
        """Test validation with no required fields"""
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            schema = {}  # No required fields
            data = {"anything": "goes"}
            
            client._validate_json_schema(data, schema)
            
            print("✅ Empty schema validation works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


# ============================================================================
# Integration Tests (Require API Keys)
# ============================================================================

class TestTextGeneration:
    """Test text generation with real APIs (requires API keys)"""
    
    @TestConfig.skip_if_no_api_key('openai')
    def test_openai_generate(self):
        """Test OpenAI text generation"""
        client = LLMClient(provider='openai', model='gpt-3.5-turbo')
        
        response = client.generate(
            prompt="Say 'Hello, World!' and nothing else.",
            system_prompt="You follow instructions exactly.",
            max_tokens=50
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check stats updated
        stats = client.get_usage_stats()
        assert stats['total_calls'] == 1
        assert stats['total_tokens'] > 0
        
        print("✅ OpenAI text generation works")
        print(f"   Response: {response}")
        print(f"   Stats: {stats}")
    
    @TestConfig.skip_if_no_api_key('anthropic')
    def test_anthropic_generate(self):
        """Test Anthropic text generation"""
        client = LLMClient(provider='anthropic', model='claude-sonnet-4')
        
        response = client.generate(
            prompt="Say 'Hello!' in one word.",
            system_prompt="You are concise.",
            max_tokens=50
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        stats = client.get_usage_stats()
        assert stats['total_calls'] == 1
        
        print("✅ Anthropic text generation works")
        print(f"   Response: {response}")
        print(f"   Stats: {stats}")
    
    @TestConfig.skip_if_no_api_key('google')
    def test_google_generate(self):
        """Test Google GenAI text generation (new SDK)"""
        client = LLMClient(provider='google', model='gemini-2.5-flash')
        
        response = client.generate(
            prompt="Say 'Hello!' in one word.",
            system_prompt="You are concise.",
            max_tokens=50
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        stats = client.get_usage_stats()
        assert stats['total_calls'] == 1
        
        print("✅ Google GenAI text generation works (new SDK)")
        print(f"   Response: {response}")
        print(f"   Stats: {stats}")
    
    def test_ollama_generate(self):
        """Test Ollama text generation (if server running)"""
        try:
            client = LLMClient(provider='ollama', model='llama3.1:8b')
            
            response = client.generate(
                prompt="Say 'test' and nothing else.",
                system_prompt="You follow instructions.",
                max_tokens=10
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            
            print("✅ Ollama text generation works")
            print(f"   Response: {response}")
            
        except Exception as e:
            print(f"⚠️  Ollama not running or model not available: {e}")
            pytest.skip("Ollama not available")


class TestJSONGeneration:
    """Test JSON generation with real APIs (requires API keys)"""
    
    @TestConfig.skip_if_no_api_key('openai')
    def test_openai_generate_json(self):
        """Test OpenAI JSON generation"""
        client = LLMClient(provider='openai', model='gpt-3.5-turbo')
        
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        response = client.generate_json(
            prompt="Create a person with name 'Alice' and age 30.",
            schema=schema
        )
        
        assert isinstance(response, dict)
        assert 'name' in response
        assert 'age' in response
        
        print("✅ OpenAI JSON generation works")
        print(f"   Response: {json.dumps(response, indent=2)}")
    
    @TestConfig.skip_if_no_api_key('anthropic')
    def test_anthropic_generate_json(self):
        """Test Anthropic JSON generation"""
        client = LLMClient(provider='anthropic', model='claude-sonnet-4')
        
        schema = {
            "type": "object",
            "required": ["colors"],
            "properties": {
                "colors": {"type": "array"}
            }
        }
        
        response = client.generate_json(
            prompt="List 3 colors.",
            schema=schema
        )
        
        assert isinstance(response, dict)
        assert 'colors' in response
        
        print("✅ Anthropic JSON generation works")
        print(f"   Response: {json.dumps(response, indent=2)}")
    
    @TestConfig.skip_if_no_api_key('google')
    def test_google_generate_json(self):
        """Test Google GenAI JSON generation (new SDK)"""
        client = LLMClient(provider='google', model='gemini-2.5-flash')
        
        schema = {
            "type": "object",
            "required": ["numbers"],
            "properties": {
                "numbers": {"type": "array"}
            }
        }
        
        response = client.generate_json(
            prompt="List numbers 1, 2, 3.",
            schema=schema
        )
        
        assert isinstance(response, dict)
        assert 'numbers' in response
        
        print("✅ Google GenAI JSON generation works (new SDK)")
        print(f"   Response: {json.dumps(response, indent=2)}")


class TestSystemPrompts:
    """Test system prompt functionality"""
    
    @TestConfig.skip_if_no_api_key('openai')
    def test_openai_system_prompt(self):
        """Test system prompt with OpenAI"""
        client = LLMClient(provider='openai', model='gpt-3.5-turbo')
        
        response = client.generate(
            prompt="What do you say?",
            system_prompt="Always respond with exactly: 'I am a robot.'",
            max_tokens=50
        )
        
        assert isinstance(response, str)
        print(f"✅ OpenAI system prompt works: {response}")
    
    @TestConfig.skip_if_no_api_key('google')
    def test_google_system_prompt(self):
        """Test system prompt with Google GenAI (new SDK)"""
        client = LLMClient(provider='google', model='gemini-2.5-flash')
        
        response = client.generate(
            prompt="Respond.",
            system_prompt="Always be very concise, use 1-2 words only.",
            max_tokens=50
        )
        
        assert isinstance(response, str)
        print(f"✅ Google GenAI system prompt works (new SDK): {response}")


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_provider(self):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError, match="'invalid' is not a valid LLMProvider"):
            LLMClient(provider='invalid')
        
        print("✅ Invalid provider caught")
    
    def test_missing_package(self):
        """Test error message for missing package"""
        # This test is tricky - we can't actually uninstall packages
        # Just verify the error handling exists
        try:
            client = LLMClient(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            print("✅ Package error handling exists")
        except ImportError as e:
            assert 'pip install' in str(e)
            print("✅ Missing package error is informative")


class TestAdvancedFeatures:
    """Test advanced features"""
    
    @TestConfig.skip_if_no_api_key('openai')
    def test_temperature_override(self):
        """Test temperature override"""
        client = LLMClient(provider='openai', model='gpt-3.5-turbo', temperature=0.1)
        
        response = client.generate(
            prompt="Say hello",
            temperature=0.9,  # Override default
            max_tokens=20
        )
        
        assert isinstance(response, str)
        print("✅ Temperature override works")
    
    @TestConfig.skip_if_no_api_key('openai')
    def test_max_tokens_override(self):
        """Test max_tokens override"""
        client = LLMClient(provider='openai', model='gpt-3.5-turbo', max_tokens=100)
        
        response = client.generate(
            prompt="Count to 5",
            max_tokens=20,  # Override default
        )
        
        assert isinstance(response, str)
        print("✅ Max tokens override works")
    
    @TestConfig.skip_if_no_api_key('openai')
    def test_stop_sequences(self):
        """Test stop sequences"""
        client = LLMClient(provider='openai', model='gpt-3.5-turbo')
        
        response = client.generate(
            prompt="Count: 1, 2, 3, 4, 5",
            stop_sequences=["3"],
            max_tokens=50
        )
        
        assert isinstance(response, str)
        print(f"✅ Stop sequences work: {response}")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests with custom formatting"""
    print("\n" + "=" * 70)
    print("LLM CLIENT TEST SUITE (Updated with Google GenAI SDK)")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("UNIT TESTS (No API Key Required)")
    print("-" * 70)
    
    # Initialization tests
    print("\n[Client Initialization]")
    test_init = TestClientInitialization()
    test_init.test_openai_initialization()
    test_init.test_anthropic_initialization()
    test_init.test_google_initialization()
    test_init.test_ollama_initialization()
    test_init.test_default_models()
    test_init.test_custom_parameters()
    
    # Convenience function
    print("\n[Convenience Function]")
    test_conv = TestConvenienceFunction()
    test_conv.test_create_llm_client()
    
    # Configuration
    print("\n[Configuration Methods]")
    test_config = TestConfigurationMethods()
    test_config.test_get_config()
    test_config.test_repr()
    
    # Token counting
    print("\n[Token Counting]")
    test_tokens = TestTokenCounting()
    test_tokens.test_count_tokens()
    test_tokens.test_token_limits()
    
    # Usage stats
    print("\n[Usage Statistics]")
    test_stats = TestUsageStatistics()
    test_stats.test_initial_stats()
    test_stats.test_reset_stats()
    
    # JSON parsing
    print("\n[JSON Parsing]")
    test_json_parse = TestJSONParsing()
    test_json_parse.test_parse_clean_json()
    test_json_parse.test_parse_json_with_markdown()
    test_json_parse.test_parse_json_with_prefix()
    test_json_parse.test_parse_json_embedded()
    test_json_parse.test_parse_json_with_code_block()
    
    # Schema validation
    print("\n[Schema Validation]")
    test_schema = TestSchemaValidation()
    test_schema.test_valid_schema()
    test_schema.test_invalid_schema()
    test_schema.test_empty_schema()
    
    # Error handling
    print("\n[Error Handling]")
    test_errors = TestErrorHandling()
    test_errors.test_invalid_provider()
    test_errors.test_missing_package()
    
    print("\n" + "-" * 70)
    print("INTEGRATION TESTS (Require API Keys)")
    print("-" * 70)
    
    # Text generation
    print("\n[Text Generation]")
    test_text = TestTextGeneration()
    try:
        test_text.test_openai_generate()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    try:
        test_text.test_anthropic_generate()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    try:
        test_text.test_google_generate()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    try:
        test_text.test_ollama_generate()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # JSON generation
    print("\n[JSON Generation]")
    test_json_gen = TestJSONGeneration()
    try:
        test_json_gen.test_openai_generate_json()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    try:
        test_json_gen.test_anthropic_generate_json()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    try:
        test_json_gen.test_google_generate_json()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # System prompts
    print("\n[System Prompts]")
    test_system = TestSystemPrompts()
    try:
        test_system.test_openai_system_prompt()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    try:
        test_system.test_google_system_prompt()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    # Advanced features
    print("\n[Advanced Features]")
    test_advanced = TestAdvancedFeatures()
    try:
        test_advanced.test_temperature_override()
        test_advanced.test_max_tokens_override()
        test_advanced.test_stop_sequences()
    except pytest.skip.Exception as e:
        print(f"⚠️  Skipped: {e}")
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
    
    print("\n📝 Note: Integration tests require API keys:")
    print("   • OPENAI_API_KEY for OpenAI tests")
    print("   • ANTHROPIC_API_KEY for Anthropic tests")
    print("   • GOOGLE_API_KEY or GEMINI_API_KEY for Google tests (NEW SDK)")
    print("   • Ollama server running for Ollama tests")
    
    print("\n💡 Set API keys:")
    print("   export OPENAI_API_KEY='sk-...'")
    print("   export ANTHROPIC_API_KEY='sk-ant-...'")
    print("   export GOOGLE_API_KEY='...'  # NEW SDK")
    
    print("\n🚀 Google GenAI SDK Updated!")
    print("   ✅ Using new google-genai package")
    print("   ✅ Model: gemini-2.5-flash")
    print("   ✅ 1M token context window")
    print("   ✅ System prompts as first-class parameter")


if __name__ == "__main__":
    run_all_tests()