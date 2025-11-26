# Knowledge Base Generator
 > AI-powered pipeline for transforming documents (pdf, docx, txt) into LLM-ready knowledge base articles

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Purpose](#purpose)
- [Functionality](#functionality)
- [Workflow](#workflow)
- [Architecture](#architecture)
- [Setup](#setup)
- [Configuration](#configuration) 
- [Usage](#usage)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Purpose

Well-structured *knowledge base* is one of the key elements of customer service systems, used not only by support agents, but also for customer-facing automatic self-service and further AI-automations as chat bots and agent assists. 

It is common for a company to have a large knowledge base documented in different formats incoherently as a legacy of inconsistent creation approach. 

**KB Generator** automates the creation of knowledge base articles from various document formats. It analyzes document content, extracts structured information, generates markdown-format articles with proper formatting, and creates rich metadata for further LLM usage (RAG agents or chat bots).

### Key benefits
- **Save Time**: Convert hours of manual KB article writing into minutes of automated processing
- **Consistency**: Ensure uniform structure, tone, and quality across all articles
- **Clean Content**: Automatically fix encoding issues, remove artifacts, and normalize formatting
- **Rich Metadata**: Automatically generate metadata, tags, and keywords
- **Multi-format Support**: Process PDFs, DOCX, and TXT files
- **Table Preservation**: Accurately extract and format tables with intelligent validation
- **Flexible AI Providers**: Support for Google Gemini, OpenAI, Anthropic Claude, and local Ollama models

---

## Functionality

### Core features
1. **Document Parsing**
   - Extract text, tables, and metadata from PDF, DOCX, and TXT files
   - Intelligent table validation to filter malformed extractions
   - Preserve document structure and formatting

2. **Content Cleaning**
   - Fix encoding issues (smart quotes, mojibake, UTF-8 errors)
   - Remove artifacts (form feeds, control characters, zero-width spaces)
   - Normalize whitespace and line breaks
   - Remove duplicate lines
   - Standardize bullet points and numbering
   - Optional header/footer removal

3. **Content Analysis**
   - AI-powered document type detection (tutorial, reference, how-to, troubleshooting, etc.)
   - Automatic section identification and outlining
   - Table placement recommendations
   - Target audience identification

4. **Article Generation**
   - Professional markdown article creation
   - Proper heading hierarchy and structure
   - Clean table formatting
   - Source attribution
   - Configurable tone and style

5. **Metadata Generation**
   - SEO-optimized titles and descriptions
   - Relevant tags and keywords
   - Difficulty level assessment
   - Reading time estimation
   - Related articles suggestions
   - Prerequisites identification

### Supported Document Types

| Type | Extensions | Table Extraction | Metadata Extraction |
|------|-----------|------------------|---------------------|
| PDF | `.pdf` | ✅ Yes | ✅ Yes (limited) |
| Word | `.docx` | ✅ Yes | ✅ Yes (full) |
| Text | `.txt` | ❌ No | ❌ No |

---

## Workflow
```
┌─────────────────────────────────────────────────────────────────┐
│                        KB GENERATOR PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

    Input Document (PDF/DOCX/TXT)
              │
              ▼
    ┌─────────────────────┐
    │  Stage 1: PARSING   │
    │  Document Parser    │
    │  • Extract text     │
    │  • Extract tables   │
    │  • Validate tables  │
    │  • Get metadata     │
    └──────────┬──────────┘
               │
               ▼
      Parsed Content + Tables
               │
               ▼
    ┌─────────────────────┐
    │ Stage 2: CLEANING   │
    │  Content Cleaner    │
    │  • Fix encoding     │
    │  • Remove artifacts │
    │  • Normalize text   │
    │  • Remove dupes     │
    └──────────┬──────────┘
               │
               ▼
       Clean Text + Tables
               │
               ▼
    ┌─────────────────────┐
    │ Stage 3: ANALYSIS   │
    │  Analysis Agent     │
    │  • Detect doc type  │
    │  • Identify sections│
    │  • Plan structure   │
    │  • Place tables     │
    └──────────┬──────────┘
               │
               ▼
       Content Plan + Structure
               │
               ▼
    ┌─────────────────────┐
    │ Stage 4: WRITING    │
    │  Writing Agent      │
    │  • Generate article │
    │  • Format markdown  │
    │  • Include tables   │
    │  • Apply style      │
    └──────────┬──────────┘
               │
               ▼
         KB Article (Markdown)
               │
               ▼
    ┌─────────────────────┐
    │ Stage 5: METADATA   │
    │  Metadata Agent     │
    │  • Generate title   │
    │  • Create tags      │
    │  • Extract keywords │
    │  • Suggest related  │
    └──────────┬──────────┘
               │
               ▼
    Output: Article + Metadata + JSON files
```

---

## Architecture

### Components

#### 1. **Document Parser** (`services/document_parser.py`)
Extracts content from various file formats with robust table validation.

**Responsibilities:**
- Parse PDF, DOCX, and TXT files
- Extract text content while preserving structure
- Identify and extract tables using `pdfplumber`
- Validate tables to filter malformed extractions
- Convert tables to markdown format
- Extract document metadata (page count, word count, etc.)

**Key Features:**
- Strict table validation to filter PDF extraction errors
- Handles empty columns, text blobs, and visual boxes
- Preserves document order in DOCX files
- Encoding detection for text files

#### 2. **Content Cleaner** (`services/content_cleaner.py`)
Cleans and normalizes extracted text for optimal LLM processing.

**Responsibilities:**
- Fix encoding issues (UTF-8 mojibake, smart quotes, Latin-1 issues)
- Remove artifacts (form feeds, control characters, BOM, zero-width spaces)
- Normalize whitespace and line breaks
- Remove consecutive duplicate lines
- Standardize bullet points and list formatting
- Optional removal of page headers/footers

**Key Features:**
- Comprehensive encoding fix database (80+ patterns)
- Configurable cleaning options
- Statistics tracking for debugging
- Conservative defaults to preserve content
- Non-destructive cleaning (validates output)

**What Gets Cleaned:**
- **Encoding Issues**: `â€™` → `'`, `Ã©` → `é`, `â€œ` → `"`
- **Artifacts**: Form feeds, control characters, zero-width spaces, BOM
- **Whitespace**: Multiple spaces → single space, max 2 consecutive newlines
- **Bullets**: `•▪▫▸▹` → `•` (normalized)
- **Duplicates**: Consecutive identical lines removed
- **Optional**: Page headers/footers ("Page X of Y")

#### 3. **Analysis Agent** (`services/analysis_agent.py`)
AI-powered content analysis and structure planning.

**Responsibilities:**
- Detect document type (tutorial, reference, concept, etc.)
- Identify target audience and difficulty level
- Extract key takeaways
- Plan article sections and hierarchy
- Recommend table placements
- Analyze content style and tone

**Key Features:**
- Multi-stage analysis with JSON output
- Intelligent section planning
- Table-to-section mapping
- Content style detection

#### 4. **Writing Agent** (`services/writing_agent.py`)
Generates professional markdown articles from content plans.

**Responsibilities:**
- Generate well-structured markdown articles
- Apply consistent formatting and style
- Place tables in appropriate locations
- Create proper heading hierarchy
- Add source attribution
- Maintain professional tone

**Key Features:**
- Template-based generation
- Configurable tone and style
- Section-by-section writing
- Table integration
- Source citation

#### 5. **Metadata Agent** (`services/metadata_agent.py`)
Creates comprehensive metadata for SEO and discoverability.

**Responsibilities:**
- Generate SEO-optimized titles
- Create meta descriptions
- Extract and suggest tags
- Identify keywords
- Estimate reading time
- Suggest related articles
- Define prerequisites

**Key Features:**
- Rich structured metadata
- SEO optimization
- Related content suggestions
- Prerequisite identification
- Comprehensive tagging

#### 6. **LLM Client** (`services/llm_client.py`)
Unified interface for multiple AI providers.

**Responsibilities:**
- Abstract provider-specific implementations
- Handle API authentication and requests
- Implement retry logic and error handling
- Parse JSON responses robustly
- Manage rate limits

**Supported Providers:**
- **Google Gemini** (gemini-2.5-flash, gemini-1.5-pro)
- **OpenAI** (gpt-4o, gpt-4o-mini, o1-mini, o1-preview)
- **Anthropic Claude** (claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus)
- **Ollama** (local models: llama3.1, qwen2.5, mistral, etc.)

### Project structure
```
kb-generator/
├── pipeline.py              # Main CLI entry point
├── config.py                # Configuration system
├── requirements.txt         # Python dependencies
├── .env                     # API keys 
├── .env.example            # Example environment file
│
├── services/                # Core service modules
│   ├── __init__.py
│   ├── models.py           # Data models and domain enums
│   ├── document_parser.py  # Document parsing & table extraction
│   ├── content_cleaner.py  # Text cleaning & normalization
│   ├── analysis_agent.py   # Content analysis & planning
│   ├── writing_agent.py    # Article generation
│   ├── metadata_agent.py   # Metadata generation
│   └── llm_client.py       # LLM provider abstraction
│
├── outputs/                 # Generated articles (auto-created)
│   └── <document-name>/
│       ├── article.md               # Final article
│       ├── article_metadata.json   # Metadata
│       ├── article_plan.json       # Content plan 
│       └── article_parsed.json     # Parsed document 
│
├── logs/                    # Execution logs (auto-created)
│   └── pipeline_YYYYMMDD_HHMMSS.log
│
└── README.md               
```

---

## Setup

### Prerequisites

- **Python 3.13+**
- API key for at least one LLM provider:
  - Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))
  - OpenAI API key ([Get it here](https://platform.openai.com/api-keys))
  - Anthropic API key ([Get it here](https://console.anthropic.com/))
  - Or Ollama installed locally ([Install Ollama](https://ollama.ai))

### Installation steps

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/kb-generator.git
cd kb-generator
```

#### 2. Create virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure API keys

Create a `.env` file in the project root:
```bash
# Copy example file
cp .env.example .env

# Edit .env and add your API keys
```

**`.env` file format:**
```env
# Choose your preferred provider and add the corresponding API key

# Google Gemini (Recommended - Fast & Free tier available)
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI (High quality, paid)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude (High quality, paid)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama (Local, free, slower)
# No API key needed - just install Ollama and pull models
```

#### 5. Verify installation
```bash
python pipeline.py --help
```

You should see the help message with available options.

---

## Configuration

KB Generator uses a centralized configuration system for maximum flexibility and control. All settings are defined in `config.py` and can be customized programmatically or via command-line arguments.

### Quick Configuration

#### Via Command Line (Easiest)
```bash
# Basic usage with defaults
python pipeline.py document.pdf

# Specify provider and model
python pipeline.py document.pdf --provider anthropic --model claude-3-5-sonnet-20241022

# Custom output and metadata
python pipeline.py document.pdf --output my-kb --author "Jane Smith" --version "2.0"

# Control what gets saved
python pipeline.py document.pdf --no-plan --no-attribution
```

#### Via Python API (Most Flexible)
```python
from config import PipelineConfig, LLMProvider, load_env_file
from pipeline import KBPipeline

# Load environment variables
load_env_file()

# Create and customize configuration
config = PipelineConfig()
config.llm.provider = LLMProvider.GOOGLE
config.llm.model = "gemini-2.5-flash"
config.output.output_dir = "my-kb"
config.output.author = "Your Name"
config.verbose = True

# Create and use pipeline
pipeline = KBPipeline(config)
result = pipeline.process_document("document.pdf")
```

---

### Configuration Sections

#### 1. LLM Configuration

Control which AI provider and model to use:
```python
# Set provider
config.llm.provider = LLMProvider.GOOGLE  # or OPENAI, ANTHROPIC, OLLAMA

# Optionally specify model (uses smart defaults if not set)
config.llm.model = "gemini-2.5-flash"

# Control generation parameters
config.llm.temperature = 0.7        # Creativity (0.0-1.0)
config.llm.max_tokens = 4000        # Max response length
config.llm.timeout = 120            # API timeout (seconds)
config.llm.max_retries = 3          # Retry failed requests
```

**Supported Providers & Default Models:**

| Provider | Default Model | Best For |
|----------|---------------|----------|
| **Google** | `gemini-2.5-flash` | Speed & cost-effectiveness |
| **OpenAI** | `gpt-4o-mini` | Quality & reliability |
| **Anthropic** | `claude-3-5-haiku-20241022` | Best quality output |
| **Ollama** | `llama3.1:8b` | Local/offline processing |

**Temperature Guidelines:**
- `0.2-0.3` - Structured output (metadata generation)
- `0.5-0.7` - Balanced (article writing) 
- `0.8-1.0` - Creative content

---

#### 2. Output Configuration

Control where and how files are generated:
```python
# Basic output settings
config.output.output_dir = "outputs"         # Output directory
config.output.create_subdirs = True          # Create folder per document
config.output.use_slug = True                # URL-friendly filenames

# Control which files to generate
config.output.generate_markdown = True       # Article (always True)
config.output.generate_metadata = True       # Metadata JSON
config.output.generate_plan = True           # Content plan JSON
config.output.generate_parsed = True         # Parsed document JSON

# Markdown options
config.output.include_frontmatter = True     # YAML frontmatter
config.output.include_source_attribution = True  # Source citation
config.output.markdown_style = "standard"    # or "obsidian"

# Article metadata
config.output.author = "Your Name"           # Author name
config.output.version = "1.0"                # Version number
```

**Output Structure:**
```
outputs/
└── document-name/
    ├── article-slug.md              # KB article
    ├── article-slug_metadata.json   # Metadata
    ├── article-slug_plan.json       # Content plan (if enabled)
    └── article-slug_parsed.json     # Parsed data (if enabled)
```

---

#### 3. Agent Configuration

Fine-tune agent behavior:
```python
# Analysis Agent
config.agent.analysis_extract_key_takeaways = True
config.agent.analysis_identify_prerequisites = True
config.agent.analysis_suggest_related_articles = True

# Writing Agent
config.agent.writing_tone = "professional"   # or "casual", "technical"
config.agent.writing_include_examples = True
config.agent.writing_max_section_length = 500  # words per section

# Metadata Agent
config.agent.metadata_generate_seo = True
config.agent.metadata_max_tags = 10
config.agent.metadata_max_keywords = 15
config.agent.metadata_max_related_articles = 8
```

---

#### 4. Parser Configuration

Control document parsing:
```python
# File size limits
config.parser.max_file_size = 100 * 1024 * 1024  # 100MB

# Table extraction
config.parser.extract_tables = True
config.parser.tables_as_markdown = True
config.parser.strict_table_validation = True

# Table validation thresholds
config.parser.min_table_rows = 2
config.parser.min_table_cols = 2
config.parser.max_empty_cell_ratio = 0.6  # Max 60% empty cells
```

---

#### 5. Cleaner Configuration

Control content cleaning behavior:
```python
# Enable/disable cleaning stage
config.cleaner.enabled = True  # Default: True

# Control individual cleaning operations
config.cleaner.remove_artifacts = True        # Remove control chars, form feeds
config.cleaner.normalize_whitespace = True    # Normalize spaces and newlines
config.cleaner.fix_encoding = True            # Fix mojibake and smart quotes
config.cleaner.remove_duplicates = True       # Remove duplicate lines
config.cleaner.clean_bullets = True           # Normalize bullet points
config.cleaner.remove_headers_footers = False # Remove page headers (aggressive)

# Statistics and limits
config.cleaner.collect_stats = False          # Track cleaning statistics
config.cleaner.max_text_length = 10_000_000   # 10MB text limit
```

**When to Adjust Cleaner Settings:**

| Scenario | Setting | Reason |
|----------|---------|--------|
| Legacy PDFs with page numbers | `remove_headers_footers = True` | Strip "Page X of Y" |
| Already clean documents | `enabled = False` | Skip cleaning for speed |
| Debugging cleaning issues | `collect_stats = True` | See what's being cleaned |
| Preserve original formatting | Set individual flags to `False` | Selective cleaning |

**Via Command Line:**
```bash
# Disable cleaning entirely
python pipeline.py document.pdf --no-cleaning

# Enable header/footer removal (aggressive)
python pipeline.py document.pdf --remove-headers

# See cleaning statistics
python pipeline.py document.pdf --cleaning-stats --verbose
```

---

### Preset Configurations

Use pre-configured setups for common scenarios:
```python
from config import (
    get_production_config,    # Google Gemini - fast, cleaning enabled
    get_quality_config,       # Claude Sonnet - best quality, stats enabled
    get_development_config,   # Ollama - local testing, full stats
    get_fast_config           # Gemini Flash - quick, aggressive cleaning
)

# Use a preset
config = get_production_config()
pipeline = KBPipeline(config)
```

**Preset Details:**

| Preset | Provider | Cleaning | Header Removal | Stats | Use Case |
|--------|----------|----------|----------------|-------|----------|
| `production` | Google | ✅ Enabled | ❌ No | ❌ No | Production use |
| `quality` | Anthropic | ✅ Enabled | ❌ No | ✅ Yes | High-quality output |
| `development` | Ollama | ✅ Enabled | ❌ No | ✅ Yes | Local testing |
| `fast` | Google | ✅ Enabled | ✅ Yes | ❌ No | Quick processing |

---

### Complete Configuration Example
```python
from config import PipelineConfig, LLMProvider, load_env_file
from pipeline import KBPipeline

# Load environment
load_env_file()

# Create configuration
config = PipelineConfig()

# LLM settings
config.llm.provider = LLMProvider.ANTHROPIC
config.llm.model = "claude-3-5-sonnet-20241022"
config.llm.temperature = 0.7

# Output settings
config.output.output_dir = "knowledge-base"
config.output.author = "Documentation Team"
config.output.version = "2.0"
config.output.generate_plan = True
config.output.generate_parsed = False

# Agent settings
config.agent.writing_tone = "professional"
config.agent.metadata_max_tags = 15

# Parser settings
config.parser.strict_table_validation = True

# Cleaner settings
config.cleaner.enabled = True
config.cleaner.remove_headers_footers = True
config.cleaner.collect_stats = True

# Pipeline settings
config.verbose = True

# Validate and use
config.validate()
pipeline = KBPipeline(config)
result = pipeline.process_document("guide.pdf")

if result.success:
    print(f"✓ Article: {result.article_path}")
    print(f"✓ Metadata: {result.metadata_path}")
```

---

### View Current Configuration
```python
from config import print_config

# Pretty-print configuration
print_config(config)
```

**Output:**
```
======================================================================
KB GENERATOR CONFIGURATION
======================================================================

📊 LLM Configuration:
  Provider: anthropic
  Model: claude-3-5-sonnet-20241022
  Temperature: 0.7
  Max Retries: 3

📄 Parser Configuration:
  Max File Size: 100 MB
  Extract Tables: True
  Strict Validation: True

🧹 Cleaner Configuration:
  Enabled: True
  Remove Artifacts: True
  Fix Encoding: True
  Remove Headers/Footers: True
  Collect Stats: True

💾 Output Configuration:
  Output Directory: knowledge-base
  Author: Documentation Team
  Version: 2.0

🤖 Agent Configuration:
  Writing Tone: professional
  Max Tags: 15

======================================================================
```

---

### Configuration Reference

| Setting | Default | Options | Description |
|---------|---------|---------|-------------|
| `llm.provider` | `GOOGLE` | `GOOGLE`, `OPENAI`, `ANTHROPIC`, `OLLAMA` | LLM provider |
| `llm.temperature` | `0.7` | `0.0-1.0` | Generation creativity |
| `output.output_dir` | `outputs` | Any path | Output directory |
| `output.author` | `None` | Any string | Article author |
| `agent.writing_tone` | `professional` | `professional`, `casual`, `technical` | Writing style |
| `parser.extract_tables` | `True` | `True`, `False` | Extract tables |
| `cleaner.enabled` | `True` | `True`, `False` | Enable content cleaning |
| `cleaner.fix_encoding` | `True` | `True`, `False` | Fix encoding issues |
| `cleaner.remove_headers_footers` | `False` | `True`, `False` | Remove page headers |

For complete configuration options, see `config.py`.

---

## Usage

### Basic usage

Generate a KB article from a single document:
```bash
python pipeline.py path/to/document.pdf
```

This will:
1. Parse the document
2. Clean the extracted content (fix encoding, remove artifacts)
3. Analyze content and create a plan
4. Generate a markdown article
5. Create metadata
6. Save outputs to `outputs/<document-name>/`

### Specify LLM provider

#### Using Google Gemini (default, recommended)
```bash
python pipeline.py document.pdf --provider google

# Specify model
python pipeline.py document.pdf --provider google --model gemini-2.5-flash
```

#### Using OpenAI
```bash
python pipeline.py document.pdf --provider openai

# Specify model
python pipeline.py document.pdf --provider openai --model gpt-4o
```

#### Using Anthropic Claude
```bash
python pipeline.py document.pdf --provider anthropic

# Specify model
python pipeline.py document.pdf --provider anthropic --model claude-3-5-sonnet-20241022
```

#### Using Ollama (local)
```bash
# First, install Ollama and pull a model
ollama pull llama3.1:8b

# Then run the pipeline
python pipeline.py document.pdf --provider ollama --model llama3.1:8b
```

**Recommended Ollama models for JSON reliability:**
- `qwen2.5:7b` - Best for structured output
- `llama3.1:8b` - Good balance of speed and quality
- `mistral:7b` - Fast and reliable

### Custom output directory
```bash
python pipeline.py document.pdf --output my-articles
```

### Process multiple documents

Process all documents in a directory:
```bash
python pipeline.py documents/ --directory
```

Process with recursion:
```bash
python pipeline.py documents/ --directory --recursive
```

Process specific file types:
```bash
python pipeline.py documents/ --directory --extensions .pdf .docx
```

### Content cleaning options

```bash
# Disable cleaning (for already clean documents)
python pipeline.py document.pdf --no-cleaning

# Enable aggressive header/footer removal
python pipeline.py document.pdf --remove-headers

# See detailed cleaning statistics
python pipeline.py document.pdf --cleaning-stats --verbose

# Combine cleaning options
python pipeline.py document.pdf --remove-headers --cleaning-stats
```

### Advanced options
```bash
# Control output files
python pipeline.py document.pdf --no-plan          # Don't save content plan
python pipeline.py document.pdf --no-attribution   # Don't include source

# Set metadata
python pipeline.py document.pdf --author "John Doe" --version "2.0"

# Verbose logging
python pipeline.py document.pdf --verbose
```

### Complete example
```bash
python pipeline.py \
  solution-brief.pdf \
  --provider google \
  --model gemini-1.5-flash \
  --output kb-articles \
  --author "Documentation Team" \
  --version "1.0" \
  --remove-headers \
  --cleaning-stats \
  --verbose
```

---

## Output files

For each processed document, the pipeline generates:

### 1. **Article markdown** (`<slug>.md`)
The final knowledge base article in markdown format with:
- YAML frontmatter (metadata)
- Structured content with proper headings
- Formatted tables
- Professional tone
- Source attribution (optional)

**Example:**
```markdown
---
title: Getting Started with Python Flask
slug: getting-started-with-python-flask
category: tutorial
difficulty: beginner
tags:
  - python
  - flask
  - web-development
---

# Getting Started with Python Flask

Flask is a lightweight web framework...
```

### 2. **Metadata JSON** (`<slug>_metadata.json`)
Comprehensive metadata including:
```json
{
  "title": "Getting Started with Python Flask",
  "slug": "getting-started-with-python-flask",
  "category": "tutorial",
  "subcategory": "getting-started",
  "tags": ["python", "flask", "web-development"],
  "keywords": ["python", "flask", "web framework"],
  "difficulty_level": "beginner",
  "estimated_reading_time": "5 minutes",
  "target_audience": "Python developers new to Flask",
  "meta_description": "Learn how to get started...",
  "prerequisites": ["Basic Python knowledge"],
  "related_articles": [...]
}
```

### 3. **Content plan JSON** (`<slug>_plan.json`) *(optional)*
The analysis and structure plan:
```json
{
  "document_type": "tutorial",
  "main_topic": "Getting started with Flask",
  "sections": [
    {
      "title": "Prerequisites",
      "level": 2,
      "summary": "Required knowledge...",
      "content_elements": ["bullet_list"],
      "estimated_length": "short"
    }
  ],
  "table_placements": [],
  "key_takeaways": [...]
}
```

### 4. **Parsed document JSON** (`<slug>_parsed.json`) *(optional)*
Raw parsed data from the document:
```json
{
  "text": "Full document text...",
  "tables": [...],
  "metadata": {
    "file_type": "pdf",
    "page_count": 10,
    "word_count": 5000
  }
}
```

---

## Troubleshooting

### Common issues

#### 1. **"No API key found"**
**Error:**
```
Error: API key required for google. Set it in .env file or pass via config.
```

**Solution:** Ensure you've created a `.env` file with the correct API key for your provider.
```bash
# Check if .env exists
ls -la .env

# Verify content
cat .env

# Should contain:
GOOGLE_API_KEY=your_actual_api_key_here
```

#### 2. **Import errors**
**Error:**
```
ImportError: attempted relative import beyond top-level package
```

**Solution:** Run the pipeline from the project root directory:
```bash
cd kb-generator
python pipeline.py document.pdf
```

#### 3. **"Failed to parse JSON response"**
**Solution:** 
- For Ollama: Use recommended models like `qwen2.5:7b` or `llama3.1:8b`
- For cloud providers: Usually a temporary issue, retry the command
- Check logs in `logs/` directory for details
- Try lowering temperature: `--temperature 0.3`

#### 4. **"Table extraction failed"**
**Solution:** The PDF might have complex formatting. The strict table validation will filter out malformed tables. Check `_parsed.json` to see what was extracted.

To disable strict validation:
```python
config = PipelineConfig()
config.parser.strict_table_validation = False
```

#### 5. **"Module not found" errors**
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

#### 6. **Ollama models produce poor output**
**Solution:** 
- Ensure you're using a model with good JSON capabilities
- Increase the model size (e.g., use 8b instead of 3b)
- Try `qwen2.5:7b` for best structured output
- Consider switching to cloud providers for production use

#### 7. **Rate limiting / 503 errors**
**Solution:**
- The pipeline automatically retries with exponential backoff
- For persistent issues, wait a few minutes and retry
- Consider using a different provider temporarily

#### 8. **Configuration not applying**
**Solution:**
- Ensure you're passing the config to the pipeline:
```python
  pipeline = KBPipeline(config)  # ✅ Correct
  pipeline = KBPipeline()        # ❌ Uses defaults
```
- Use `print_config(config)` to verify settings
- Check that `.env` file is loaded: `load_env_file()`

#### 9. **Cleaning seems too aggressive** 
**Problem:** Content is being removed or over-normalized.

**Solution:**
```bash
# Disable cleaning entirely
python pipeline.py document.pdf --no-cleaning

# Or disable specific operations programmatically
config.cleaner.remove_headers_footers = False
config.cleaner.remove_duplicates = False
```

#### 10. **Want to see what's being cleaned** 
**Problem:** Need to verify cleaning is working correctly.

**Solution:**
```bash
# Enable statistics
python pipeline.py document.pdf --cleaning-stats --verbose
```

**Output will show:**
```
[Stage 2/5] Cleaning content...
Cleaned 15,234 → 14,987 chars
Applied: 12 encoding fixes, 3 duplicates, 5 artifacts
✓ Content cleaned successfully
```

#### 11. **Encoding issues in final article** 
**Problem:** Still seeing `â€™` or `Ã©` in output.

**Solution:**
- Cleaning should be enabled by default
- Check if cleaning is disabled: `python pipeline.py document.pdf --verbose` (should show Stage 2)
- Try processing with stats: `python pipeline.py document.pdf --cleaning-stats --verbose`
- If issues persist, the encoding may be in a non-standard format

---


### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/kb-generator.git
cd kb-generator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make changes and test
python pipeline.py test-document.pdf --verbose
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---