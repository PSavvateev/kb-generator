"""
Configuration module for KB Generator Pipeline

Defines all configuration settings, enums, and dataclasses used throughout the pipeline.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
from pathlib import Path
from services.models import DocumentType, DifficultyLevel

# ============================================================================
# Enums
# ============================================================================

class LLMProvider(Enum):
    """Supported LLM providers"""
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: LLMProvider = LLMProvider.GOOGLE
    model: Optional[str] = None  
    
    # API keys (loaded from environment)
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    
    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    
    # Model defaults for each provider
    default_models: Dict[LLMProvider, str] = field(default_factory=lambda: {
        LLMProvider.GOOGLE: "gemini-2.5-flash",
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
        LLMProvider.OLLAMA: "llama3.1:8b"
    })

    # Token limits
    token_limits: Dict[str, int] = field(default_factory=lambda: {
        "gpt-4": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-sonnet-4": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-5-opus-20241022": 200000,
        "gemini-2.5-flash": 1000000,
        "gemini-1.5-flash": 1000000,
        "gemini-1.5-pro": 2000000,
        "gemini-pro": 32768,
        "llama3.1:8b": 8192,
        "llama3.1:70b": 8192,
        "qwen2.5:7b": 8192,
        "mistral:7b": 8192,
    })

    # Required packages
    required_packages: Dict[LLMProvider, str] = field(default_factory=lambda: {
        LLMProvider.OPENAI: "openai",
        LLMProvider.ANTHROPIC: "anthropic",
        LLMProvider.GOOGLE: "google-genai",
        LLMProvider.OLLAMA: "ollama"
    })

    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = 4000
    timeout: int = 120
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    def get_model(self) -> str:
        """Get the model to use for the configured provider"""
        if self.model:
            return self.model
        
        default = self.default_models.get(self.provider)

        if default is None:
            raise ValueError(
                f"No default model configured for provider: {self.provider.value}. "
                f"Please set model explicitly or add to default_models."
            )

        return default
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key for the configured provider"""
        if self.provider == LLMProvider.GOOGLE:
            return self.google_api_key
        elif self.provider == LLMProvider.OPENAI:
            return self.openai_api_key
        elif self.provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        return None
    
    def validate(self) -> bool:
        """Validate configuration"""
        # Check API key
        if self.provider != LLMProvider.OLLAMA:
            api_key = self.get_api_key()
            if not api_key:
                raise ValueError(
                    f"API key not found for provider: {self.provider.value}. "
                    f"Please set it in your .env file."
                )
        
        # Validate temperature range
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got: {self.temperature}")
        
        # Validate max_tokens
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got: {self.max_tokens}")
        
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got: {self.timeout}")
        
        return True


@dataclass
class ParserConfig:
    """Document parser configuration"""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    extract_tables: bool = True
    tables_as_markdown: bool = True
    strict_table_validation: bool = True
    
    # Table validation parameters
    min_table_rows: int = 2
    min_table_cols: int = 2
    max_empty_cell_ratio: float = 0.6  # Reject if >60% cells are empty


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "outputs"
    create_subdirs: bool = True  # Create subdirectory per document
    
    # File naming
    use_slug: bool = True  # Use slug for filenames instead of original name
    
    # Output files to generate
    generate_markdown: bool = True
    generate_metadata: bool = True
    generate_plan: bool = True
    generate_parsed: bool = True  # Save parsed document JSON
    
    # Markdown configuration
    include_frontmatter: bool = True
    include_source_attribution: bool = True
    markdown_style: str = "standard"  # "standard" or "obsidian"
    
    # Metadata
    author: Optional[str] = None
    version: str = "1.0"
    
    def get_output_path(self, base_name: str) -> Path:
        """Get output directory path for a document"""
        output_path = Path(self.output_dir)
        if self.create_subdirs:
            output_path = output_path / base_name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    
    # Analysis Agent
    analysis_extract_key_takeaways: bool = True
    analysis_identify_prerequisites: bool = True
    analysis_suggest_related_articles: bool = True
    
    # Writing Agent
    writing_tone: str = "professional"  # professional, casual, technical
    writing_include_examples: bool = True
    writing_max_section_length: int = 500  # words per section
    
    # Metadata Agent
    metadata_generate_seo: bool = True
    metadata_max_tags: int = 10
    metadata_max_keywords: int = 15
    metadata_max_related_articles: int = 8


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    
    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Pipeline behavior
    verbose: bool = False
    dry_run: bool = False  # Parse only, don't generate
    
    # Logging
    log_to_file: bool = True
    log_dir: str = "logs"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    @classmethod
    def from_args(cls, args) -> 'PipelineConfig':
        """Create configuration from command-line arguments"""
        config = cls()
        
        # LLM configuration
        if hasattr(args, 'provider') and args.provider:
            config.llm.provider = LLMProvider(args.provider)
        if hasattr(args, 'model') and args.model:
            config.llm.model = args.model
        if hasattr(args, 'temperature') and args.temperature is not None:
            config.llm.temperature = args.temperature
        
        # Output configuration
        if hasattr(args, 'output_dir') and args.output_dir:
            config.output.output_dir = args.output_dir
        if hasattr(args, 'author') and args.author:
            config.output.author = args.author
        if hasattr(args, 'version') and args.version:
            config.output.version = args.version
        if hasattr(args, 'no_source_attribution') and args.no_source_attribution:
            config.output.include_source_attribution = False
        
        # Pipeline behavior
        if hasattr(args, 'verbose') and args.verbose:
            config.verbose = True
            config.log_level = "DEBUG"
        if hasattr(args, 'dry_run') and args.dry_run:
            config.dry_run = True
        
        return config
    
    def validate(self) -> bool:
        """Validate all configuration"""
        self.llm.validate()
        
        # Validate output directory
        Path(self.output.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate log directory
        if self.log_to_file:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        return True


# ============================================================================
# Preset Configurations
# ============================================================================

def get_production_config() -> PipelineConfig:
    """Production-ready configuration with Google Gemini"""
    config = PipelineConfig()
    config.llm.provider = LLMProvider.GOOGLE
    config.llm.model = "gemini-1.5-flash"
    config.llm.temperature = 0.7
    config.verbose = False
    config.log_level = "INFO"
    return config


def get_quality_config() -> PipelineConfig:
    """High-quality configuration with Claude Sonnet"""
    config = PipelineConfig()
    config.llm.provider = LLMProvider.ANTHROPIC
    config.llm.model = "claude-3-5-sonnet-20241022"
    config.llm.temperature = 0.8
    config.verbose = True
    config.log_level = "DEBUG"
    return config


def get_development_config() -> PipelineConfig:
    """Development configuration with local Ollama"""
    config = PipelineConfig()
    config.llm.provider = LLMProvider.OLLAMA
    config.llm.model = "qwen2.5:7b"
    config.verbose = True
    config.log_level = "DEBUG"
    config.output.generate_parsed = True
    config.output.generate_plan = True
    return config


def get_fast_config() -> PipelineConfig:
    """Fast configuration for quick testing"""
    config = PipelineConfig()
    config.llm.provider = LLMProvider.GOOGLE
    config.llm.model = "gemini-1.5-flash"
    config.llm.temperature = 0.5
    config.agent.analysis_suggest_related_articles = False
    config.verbose = False
    return config


# ============================================================================
# Helper Functions
# ============================================================================

def load_env_file(env_path: str = ".env", verbose: bool = False) -> bool:
    """
    Load environment variables from .env file
    
    Args:
        env_path: Path to .env file
        verbose: Print status messages
        
    Returns:
        True if .env file was loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
        
        # Check if file exists
        if not Path(env_path).exists():
            if verbose:
                print(f"Note: {env_path} file not found. Using system environment variables.")
            return False
        
        load_dotenv(env_path)
        if verbose:
            print(f"✓ Loaded environment variables from {env_path}")
        return True
        
    except ImportError:
        if verbose:
            print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
            print("Using system environment variables instead.")
        return False

def print_config(config: PipelineConfig) -> None:
    """Pretty print configuration"""
    print("\n" + "="*70)
    print("KB GENERATOR CONFIGURATION")
    print("="*70)
    
    # LLM Configuration
    if hasattr(config, 'llm') and config.llm:
        print("\n📊 LLM Configuration:")
        print(f"  Provider: {config.llm.provider.value}")
        try:
            print(f"  Model: {config.llm.get_model()}")
        except ValueError as e:
            print(f"  Model: Not set ({e})")
        print(f"  Temperature: {config.llm.temperature}")
        print(f"  Max Tokens: {config.llm.max_tokens}")
        print(f"  Max Retries: {config.llm.max_retries}")
    
    # Parser Configuration
    if hasattr(config, 'parser') and config.parser:
        print("\n📄 Parser Configuration:")
        print(f"  Max File Size: {config.parser.max_file_size / (1024*1024):.0f} MB")
        print(f"  Extract Tables: {config.parser.extract_tables}")
        print(f"  Strict Validation: {config.parser.strict_table_validation}")
    
    # Output Configuration
    if hasattr(config, 'output') and config.output:
        print("\n💾 Output Configuration:")
        print(f"  Output Directory: {config.output.output_dir}")
        print(f"  Create Subdirs: {config.output.create_subdirs}")
        print(f"  Include Source: {config.output.include_source_attribution}")
        print(f"  Author: {config.output.author or 'Not set'}")
        print(f"  Version: {config.output.version}")
    
    # Agent Configuration
    if hasattr(config, 'agent') and config.agent:
        print("\n🤖 Agent Configuration:")
        print(f"  Writing Tone: {config.agent.writing_tone}")
        print(f"  Generate SEO: {config.agent.metadata_generate_seo}")
        print(f"  Max Tags: {config.agent.metadata_max_tags}")
        print(f"  Max Keywords: {config.agent.metadata_max_keywords}")
    
    # Pipeline Configuration
    print("\n🔧 Pipeline Configuration:")
    print(f"  Verbose: {config.verbose}")
    print(f"  Dry Run: {config.dry_run}")
    print(f"  Log Level: {config.log_level}")
    print(f"  Log to File: {config.log_to_file}")
    if config.log_to_file:
        print(f"  Log Directory: {config.log_dir}")
    
    print("\n" + "="*70 + "\n")

# ============================================================================
# Export
# ============================================================================

__all__ = [
    'LLMProvider',
    'DocumentType',
    'DifficultyLevel',
    'LLMConfig',
    'ParserConfig',
    'OutputConfig',
    'AgentConfig',
    'PipelineConfig',
    'get_production_config',
    'get_quality_config',
    'get_development_config',
    'get_fast_config',
    'load_env_file',
    'print_config',
]