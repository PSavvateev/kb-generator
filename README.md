# Knowledge Base Generator
 > AI-powered pipeline for transforming documents (pdf, docx, txt) into LLM-ready knowledge base articles

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Purpose

Well-structured *knowledge base* is one of the key elements of customer service systems, used not only by support agents, but also for customer-facing automatic self-service and further AI-automations as chat bots and agent assists. 

It is common for a company to have a large knowledge base documented in different formats incoherently as a legacy of inconsistent creation approach. 

**KB Generator** automates the creation of knowledge base articles from various document formats. It analyzes document content, extracts structured information, generates markdown-format articles with proper formatting, and creates rich metadata for further LLM usage (RAG agents or chat bots).


### Key benefits
- **Save Time**: Convert hours of manual KB article writing into minutes of automated processing
- **Consistency**: Ensure uniform structure, tone, and quality across all articles
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

2. **Content Analysis**
   - AI-powered document type detection (tutorial, reference, how-to, troubleshooting, etc.)
   - Automatic section identification and outlining
   - Table placement recommendations
   - Target audience identification

3. **Article Generation**
   - Professional markdown article creation
   - Proper heading hierarchy and structure
   - Clean table formatting
   - Source attribution
   - Configurable tone and style

4. **Metadata Generation**
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
    │ Stage 2: ANALYSIS   │
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
    │ Stage 3: WRITING    │
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
    │ Stage 4: METADATA   │
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

## Architechture

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

#### 2. **Analysis Agent** (`services/analysis_agent.py`)
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

#### 3. **Writing Agent** (`services/writing_agent.py`)
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

#### 4. **Metadata Agent** (`services/metadata_agent.py`)
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

#### 5. **LLM Client** (`services/llm_client.py`)
Unified interface for multiple AI providers.

**Responsibilities:**
- Abstract provider-specific implementations
- Handle API authentication and requests
- Implement retry logic and error handling
- Parse JSON responses robustly
- Manage rate limits

**Supported Providers:**
- **Google Gemini** (gemini-1.5-flash, gemini-1.5-pro)
- **OpenAI** (gpt-4o, gpt-4o-mini, o1-mini, o1-preview)
- **Anthropic Claude** (claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus)
- **Ollama** (local models: llama3.1, qwen2.5, mistral, etc.)

### Project structure

```
kb-generator/
├── pipeline.py              # Main CLI entry point
├── config.py                # Configuration dataclasses
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not in git)
├── .env.example            # Example environment file
│
├── services/                # Core service modules
│   ├── __init__.py
│   ├── document_parser.py  # Document parsing & table extraction
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
└── README.md               # This file
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
git clone 
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

## Usage

### Basic usage

Generate a KB article from a single document:

```bash
python pipeline.py path/to/document.pdf
```

This will:
1. Parse the document
2. Analyze content and create a plan
3. Generate a markdown article
4. Create metadata
5. Save outputs to `outputs/<document-name>/`

### Specify LLM provider

#### Using Google Gemini (default, recommended)

```bash
python pipeline.py document.pdf --provider google

# Specify model
python pipeline.py document.pdf --provider google --model gemini-1.5-pro
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
python pipeline.py document.pdf --output-dir my-articles
```

### Process multiple documents

Process all documents in a directory:

```bash
python pipeline.py documents/ --directory
```

Process specific file types:

```bash
python pipeline.py documents/ --directory --pattern "*.pdf"
```

### Advanced options

```bash
# Disable source attribution in articles
python pipeline.py document.pdf --no-source-attribution

# Use custom author name
python pipeline.py document.pdf --author "John Doe"

# Set custom version
python pipeline.py document.pdf --version "2.0"

# Verbose logging
python pipeline.py document.pdf --verbose

# Dry run (parse only, no generation)
python pipeline.py document.pdf --dry-run
```

### Complete example

```bash
python pipeline.py \
  solution-brief.pdf \
  --provider google \
  --model gemini-1.5-flash \
  --output-dir kb-articles \
  --author "Documentation Team" \
  --version "1.0" \
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

### 2. **Metadata JSON** (`<slug>_metadata.json`)
Comprehensive metadata including:
```json
{
  "title": "Article Title",
  "slug": "article-slug",
  "category": "tutorial",
  "subcategory": "getting-started",
  "tags": ["tag1", "tag2"],
  "keywords": ["keyword1", "keyword2"],
  "difficulty_level": "beginner",
  "estimated_reading_time": "5 minutes",
  "target_audience": "Description...",
  "meta_description": "SEO description...",
  "prerequisites": ["prerequisite1"],
  "related_articles": [...]
}
```

### 3. **Content plan JSON** (`<slug>_plan.json`)
The analysis and structure plan:
```json
{
  "document_type": "tutorial",
  "main_topic": "Topic description",
  "sections": [...],
  "table_placements": [...],
  "key_takeaways": [...]
}
```

### 4. **Parsed document JSON** (`<slug>_parsed.json`)
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
**Solution:** Ensure you've created a `.env` file with the correct API key for your provider.

```bash
# Check if .env exists
ls -la .env

# Verify content
cat .env
```

#### 2. **"Failed to parse JSON response"**
**Solution:** 
- For Ollama: Use recommended models like `qwen2.5:7b` or `llama3.1:8b`
- For cloud providers: Usually a temporary issue, retry the command
- Check logs in `logs/` directory for details

#### 3. **"Table extraction failed"**
**Solution:** The PDF might have complex formatting. The strict table validation will filter out malformed tables. Check `_parsed.json` to see what was extracted.

#### 4. **"Module not found" errors**
**Solution:** Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

#### 5. **Ollama models produce poor output**
**Solution:** 
- Ensure you're using a model with good JSON capabilities
- Increase the model size (e.g., use 8b instead of 3b)
- Consider switching to cloud providers for production use

#### 6. **Output markdown looks wrong in Obsidian**
**Solution:** Obsidian has different markdown rendering. The markdown is correct for standard parsers (GitHub, VS Code). Future versions may include Obsidian-specific formatting.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---







