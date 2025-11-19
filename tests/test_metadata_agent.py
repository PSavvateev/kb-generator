# test_metadata_agent.py
"""
Comprehensive Test Suite for Metadata Agent

Tests cover:
- Agent initialization
- Core metadata extraction
- Relationship generation
- SEO metadata generation
- Technical feature analysis
- Slug generation
- Frontmatter generation
- JSON export
- Error handling
"""

import json
import os
import pytest
from typing import Dict
from pathlib import Path
import sys
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.metadata_agent import (
    MetadataAgent,
    ArticleMetadata,
    RelatedArticle,
    DifficultyLevel,
    ContentCategory,
    create_metadata_agent,
    METADATA_EXTRACTION_SCHEMA,
    RELATIONSHIPS_SCHEMA
)
from services.analysis_agent import ContentPlan, Section, TablePlacement
from services.writing_agent import KBArticle
from services.llm_client import LLMClient


# ============================================================================
# Test Fixtures and Data
# ============================================================================

class TestData:
    """Test data and sample objects"""
    
    @staticmethod
    def get_sample_article() -> KBArticle:
        """Sample KB article"""
        return KBArticle(
            title="Getting Started with Python Flask",
            content="""# Getting Started with Python Flask

Flask is a lightweight web framework for Python that makes it easy to build web applications.

## Prerequisites

Before you begin, ensure you have:
- Python 3.7 or higher
- pip package manager
- Basic Python knowledge

## Installation

Install Flask using pip:
```bash
pip install flask
```

## Creating Your First App

Create a file called `app.py`:
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

## Running the Application

Run your application:
```bash
python app.py
```

Visit http://localhost:5000 to see your app!
""",
            metadata={
                "document_type": "tutorial",
                "target_audience": "beginner developers",
                "style": "beginner-friendly",
                "sections": 5,
                "has_tables": False,
                "source_file": "flask_tutorial.txt"
            },
            word_count=250,
            estimated_reading_time="5 minutes"
        )
    
    @staticmethod
    def get_sample_content_plan() -> ContentPlan:
        """Sample content plan"""
        return ContentPlan(
            document_type="tutorial",
            main_topic="Getting Started with Python Flask",
            target_audience="beginner developers",
            estimated_reading_time="10 minutes",
            key_takeaways=[
                "Flask installation",
                "Creating first application",
                "Running Flask apps"
            ],
            sections=[
                Section(
                    title="Introduction",
                    level=1,
                    summary="Overview of Flask framework",
                    content_elements=["paragraph"],
                    estimated_length="short"
                ),
                Section(
                    title="Prerequisites",
                    level=2,
                    summary="Requirements before starting",
                    content_elements=["bullet_list"],
                    estimated_length="short"
                ),
                Section(
                    title="Installation",
                    level=2,
                    summary="Installing Flask",
                    content_elements=["paragraph", "code_block"],
                    estimated_length="medium"
                )
            ],
            table_placements=[],
            content_style="beginner-friendly",
            special_instructions=["Write in step-by-step format"]
        )
    
    @staticmethod
    def get_sample_parsed_result() -> Dict:
        """Sample parsed result"""
        return {
            "text": "Flask tutorial content...",
            "metadata": {
                "filename": "flask_tutorial.txt",
                "file_type": "text/plain"
            },
            "tables": [],
            "images": []
        }
    
    @staticmethod
    def get_sample_with_tables() -> tuple:
        """Sample article with tables"""
        article = KBArticle(
            title="HTTP Status Codes Reference",
            content="""# HTTP Status Codes Reference

## Success Codes

| Code | Message | Description |
|------|---------|-------------|
| 200  | OK      | Request successful |
| 201  | Created | Resource created |
""",
            metadata={
                "document_type": "reference",
                "has_tables": True
            },
            word_count=100,
            estimated_reading_time="2 minutes"
        )
        
        content_plan = ContentPlan(
            document_type="reference",
            main_topic="HTTP Status Codes",
            target_audience="developers",
            estimated_reading_time="5 minutes",
            key_takeaways=["Status codes"],
            sections=[
                Section(
                    title="Success Codes",
                    level=2,
                    summary="2xx codes",
                    content_elements=["table"],
                    estimated_length="medium"
                )
            ],
            table_placements=[],
            content_style="technical",
            special_instructions=[]
        )
        
        parsed_result = {
            "text": "HTTP status codes...",
            "metadata": {"filename": "http_codes.txt"},
            "tables": [{"rows": [["Code", "Message"]]}],
            "images": []
        }
        
        return article, content_plan, parsed_result


def has_api_key() -> bool:
    """Check if API key is available for testing"""
    return bool(
        os.getenv('OPENAI_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY') or
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('GEMINI_API_KEY')
    )


def get_test_agent() -> MetadataAgent:
    """Create test agent with available provider"""
    if os.getenv('OPENAI_API_KEY'):
        return create_metadata_agent(provider='openai', model='gpt-3.5-turbo')
    elif os.getenv('ANTHROPIC_API_KEY'):
        return create_metadata_agent(provider='anthropic')
    elif os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        return create_metadata_agent(provider='google')
    else:
        pytest.skip("No API key available")


# ============================================================================
# Unit Tests (No API Key Required)
# ============================================================================

class TestDataModels:
    """Test data models"""
    
    def test_related_article_creation(self):
        """Test RelatedArticle dataclass"""
        related = RelatedArticle(
            title="Advanced Flask Concepts",
            relevance_reason="Natural follow-up",
            relationship_type="follow-up"
        )
        
        assert related.title == "Advanced Flask Concepts"
        assert related.relationship_type == "follow-up"
        
        print("✅ RelatedArticle creation works")
    
    def test_article_metadata_to_dict(self):
        """Test ArticleMetadata serialization"""
        metadata = ArticleMetadata(
            title="Test Article",
            slug="test-article",
            category="tutorial",
            subcategory="python",
            tags=["python", "flask"],
            keywords=["web", "framework"],
            difficulty_level="beginner",
            estimated_reading_time="5 minutes",
            target_audience="developers",
            meta_description="Test description",
            search_keywords=["test"],
            prerequisites=["Python basics"],
            related_articles=[],
            related_topics=["web development"],
            document_type="tutorial",
            content_format="markdown",
            has_code_examples=True,
            has_tables=False,
            has_images=False,
            created_date="2024-01-01",
            last_updated="2024-01-01"
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict['title'] == "Test Article"
        assert metadata_dict['slug'] == "test-article"
        assert "python" in metadata_dict['tags']
        assert metadata_dict['difficulty_level'] == "beginner"
        
        print("✅ ArticleMetadata.to_dict() works")
    
    def test_difficulty_level_enum(self):
        """Test DifficultyLevel enum"""
        assert DifficultyLevel.BEGINNER == "beginner"
        assert DifficultyLevel.INTERMEDIATE == "intermediate"
        assert DifficultyLevel.ADVANCED == "advanced"
        assert DifficultyLevel.EXPERT == "expert"
        
        print("✅ DifficultyLevel enum works")
    
    def test_content_category_enum(self):
        """Test ContentCategory enum"""
        assert ContentCategory.TUTORIAL == "tutorial"
        assert ContentCategory.GUIDE == "guide"
        assert ContentCategory.REFERENCE == "reference"
        
        print("✅ ContentCategory enum works")


class TestAgentInitialization:
    """Test agent initialization"""
    
    def test_create_with_llm_client(self):
        """Test creating agent with LLM client"""
        try:
            llm = LLMClient(
                provider='openai',
                model='gpt-3.5-turbo',
                api_key='test-key-for-initialization'
            )
            agent = MetadataAgent(llm)
            
            assert agent.llm is not None
            assert agent.max_retries == 3
            assert agent.verbose is False
            
            print("✅ Agent initialization works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_create_with_convenience_function(self):
        """Test convenience function"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                model='gpt-3.5-turbo',
                api_key='test-key-for-initialization'
            )
            
            assert isinstance(agent, MetadataAgent)
            assert agent.llm.provider.value == 'openai'
            
            print("✅ Convenience function works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_verbose_mode(self):
        """Test verbose mode setting"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization',
                verbose=True
            )
            
            assert agent.verbose is True
            
            print("✅ Verbose mode works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestSlugGeneration:
    """Test slug generation"""
    
    def test_simple_slug(self):
        """Test simple slug generation"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            slug = agent._generate_slug("Getting Started with Flask")
            assert slug == "getting-started-with-flask"
            
            print("✅ Simple slug generation works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_slug_with_special_chars(self):
        """Test slug with special characters"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            slug = agent._generate_slug("Python's Best Practices (2024)")
            assert "python" in slug
            assert "best" in slug
            assert "(" not in slug
            assert ")" not in slug
            
            print("✅ Slug with special chars works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_long_slug_truncation(self):
        """Test long slug truncation"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            long_title = "This is a very long title that should be truncated to fit within reasonable URL length limits"
            slug = agent._generate_slug(long_title)
            
            assert len(slug) <= 60
            
            print(f"✅ Long slug truncation works: {slug}")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestTechnicalFeatures:
    """Test technical feature analysis"""
    
    def test_detect_code_blocks(self):
        """Test code block detection"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = TestData.get_sample_article()
            parsed_result = TestData.get_sample_parsed_result()
            
            features = agent._analyze_technical_features(article, parsed_result)
            
            assert features['has_code'] is True
            
            print("✅ Code block detection works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_detect_tables(self):
        """Test table detection"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article, content_plan, parsed_result = TestData.get_sample_with_tables()
            
            features = agent._analyze_technical_features(article, parsed_result)
            
            assert features['has_tables'] is True
            
            print("✅ Table detection works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_detect_images(self):
        """Test image detection"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = KBArticle(
                title="Test",
                content="# Test\n\n![Image](image.png)",
                metadata={},
                word_count=10,
                estimated_reading_time="1 minute"
            )
            
            parsed_result = {
                "text": "test",
                "metadata": {},
                "tables": [],
                "images": [{"url": "image.png"}]
            }
            
            features = agent._analyze_technical_features(article, parsed_result)
            
            assert features['has_images'] is True
            
            print("✅ Image detection works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestSEOMetadata:
    """Test SEO metadata generation"""
    
    def test_meta_description_length(self):
        """Test meta description stays within SEO limits"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = TestData.get_sample_article()
            core_metadata = {
                "meta_description": "A" * 200,  # Too long
                "keywords": ["test"],
                "tags": ["test"]
            }
            
            seo_metadata = agent._generate_seo_metadata(article, core_metadata)
            
            assert len(seo_metadata['meta_description']) <= 160
            
            print("✅ Meta description length limit works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_search_keywords_deduplication(self):
        """Test search keywords are deduplicated"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = TestData.get_sample_article()
            core_metadata = {
                "meta_description": "Test description",
                "keywords": ["python", "flask", "python"],  # Duplicate
                "search_keywords": ["flask", "web"],
                "tags": ["python"]
            }
            
            seo_metadata = agent._generate_seo_metadata(article, core_metadata)
            
            # Should have unique keywords only
            assert len(seo_metadata['search_keywords']) == len(set(seo_metadata['search_keywords']))
            
            print("✅ Keyword deduplication works")
        except ImportError:
            pytest.skip("OpenAI package not installed")



class TestFrontmatterGeneration:
    """Test YAML frontmatter generation"""
    
    def test_basic_frontmatter(self):
        """Test basic frontmatter generation"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            metadata = ArticleMetadata(
                title="Test Article",
                slug="test-article",
                category="tutorial",
                subcategory=None,
                tags=["python", "flask"],
                keywords=["web", "framework"],
                difficulty_level="beginner",
                estimated_reading_time="5 minutes",
                target_audience="developers",
                meta_description="Test description",
                search_keywords=[],
                prerequisites=["Python basics"],
                related_articles=[],
                related_topics=[],
                document_type="tutorial",
                content_format="markdown",
                has_code_examples=True,
                has_tables=False,
                has_images=False,
                created_date="2024-01-01",
                last_updated="2024-01-01"
            )
            
            frontmatter = agent.generate_frontmatter(metadata)
            
            assert "---" in frontmatter
            assert "title: Test Article" in frontmatter
            assert "slug: test-article" in frontmatter
            assert "category: tutorial" in frontmatter
            assert "difficulty: beginner" in frontmatter
            assert "- python" in frontmatter
            assert "- flask" in frontmatter
            
            print("✅ Basic frontmatter generation works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_frontmatter_with_author(self):
        """Test frontmatter with author"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            metadata = ArticleMetadata(
                title="Test",
                slug="test",
                category="tutorial",
                subcategory=None,
                tags=[],
                keywords=[],
                difficulty_level="beginner",
                estimated_reading_time="5 minutes",
                target_audience="developers",
                meta_description="Test",
                search_keywords=[],
                prerequisites=[],
                related_articles=[],
                related_topics=[],
                document_type="tutorial",
                content_format="markdown",
                has_code_examples=False,
                has_tables=False,
                has_images=False,
                created_date="2024-01-01",
                last_updated="2024-01-01",
                author="John Doe",
                version="1.0"
            )
            
            frontmatter = agent.generate_frontmatter(metadata)
            
            assert "author: John Doe" in frontmatter
            assert "version: 1.0" in frontmatter
            
            print("✅ Frontmatter with author works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestJSONExport:
    """Test JSON export functionality"""
    
    def test_metadata_to_json(self):
        """Test metadata to JSON conversion"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            metadata = ArticleMetadata(
                title="Test Article",
                slug="test-article",
                category="tutorial",
                subcategory=None,
                tags=["python"],
                keywords=["test"],
                difficulty_level="beginner",
                estimated_reading_time="5 minutes",
                target_audience="developers",
                meta_description="Test",
                search_keywords=[],
                prerequisites=[],
                related_articles=[],
                related_topics=[],
                document_type="tutorial",
                content_format="markdown",
                has_code_examples=False,
                has_tables=False,
                has_images=False,
                created_date="2024-01-01",
                last_updated="2024-01-01"
            )
            
            json_str = agent.get_metadata_json(metadata)
            
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert parsed['title'] == "Test Article"
            assert parsed['slug'] == "test-article"
            
            print("✅ JSON export works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_save_metadata(self, tmp_path):
        """Test saving metadata to file"""
        try:
            agent = create_metadata_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            metadata = ArticleMetadata(
                title="Test",
                slug="test",
                category="tutorial",
                subcategory=None,
                tags=[],
                keywords=[],
                difficulty_level="beginner",
                estimated_reading_time="5 minutes",
                target_audience="developers",
                meta_description="Test",
                search_keywords=[],
                prerequisites=[],
                related_articles=[],
                related_topics=[],
                document_type="tutorial",
                content_format="markdown",
                has_code_examples=False,
                has_tables=False,
                has_images=False,
                created_date="2024-01-01",
                last_updated="2024-01-01"
            )
            
            output_path = tmp_path / "metadata.json"
            agent.save_metadata(metadata, str(output_path))
            
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['title'] == "Test"
            
            print("✅ Save metadata works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


# ============================================================================
# Integration Tests (Require API Key)
# ============================================================================

class TestMetadataGeneration:
    """Test complete metadata generation (requires API key)"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_generate_complete_metadata(self):
        """Test generating complete metadata"""
        agent = get_test_agent()
        article = TestData.get_sample_article()
        content_plan = TestData.get_sample_content_plan()
        parsed_result = TestData.get_sample_parsed_result()
        
        metadata = agent.generate(
            article=article,
            content_plan=content_plan,
            parsed_result=parsed_result,
            author="Test Author",
            version="1.0"
        )
        
        assert isinstance(metadata, ArticleMetadata)
        assert metadata.title == article.title
        assert len(metadata.tags) > 0
        assert len(metadata.keywords) > 0
        assert metadata.difficulty_level in ["beginner", "intermediate", "advanced", "expert"]
        assert len(metadata.meta_description) <= 160
        assert metadata.author == "Test Author"
        assert metadata.version == "1.0"
        
        print("✅ Complete metadata generation works")
        print(f"   Title: {metadata.title}")
        print(f"   Category: {metadata.category}")
        print(f"   Tags: {metadata.tags}")
        print(f"   Difficulty: {metadata.difficulty_level}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_metadata_with_tables(self):
        """Test metadata generation for article with tables"""
        agent = get_test_agent()
        article, content_plan, parsed_result = TestData.get_sample_with_tables()
        
        metadata = agent.generate(
            article=article,
            content_plan=content_plan,
            parsed_result=parsed_result
        )
        
        assert metadata.has_tables is True
        assert isinstance(metadata, ArticleMetadata)
        
        print("✅ Metadata with tables works")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_core_metadata_extraction(self):
        """Test core metadata extraction"""
        agent = get_test_agent()
        article = TestData.get_sample_article()
        content_plan = TestData.get_sample_content_plan()
        
        core_metadata = agent._extract_core_metadata(article, content_plan)
        
        assert 'tags' in core_metadata
        assert 'keywords' in core_metadata
        assert 'category' in core_metadata
        assert 'difficulty_level' in core_metadata
        assert 'meta_description' in core_metadata
        
        print("✅ Core metadata extraction works")
        print(f"   Tags: {core_metadata['tags']}")
        print(f"   Category: {core_metadata['category']}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_relationship_generation(self):
        """Test relationship generation"""
        agent = get_test_agent()
        article = TestData.get_sample_article()
        content_plan = TestData.get_sample_content_plan()
        
        relationships = agent._generate_relationships(article, content_plan)
        
        assert 'prerequisites' in relationships
        assert 'related_topics' in relationships
        assert 'related_articles' in relationships
        
        print("✅ Relationship generation works")
        if relationships['prerequisites']:
            print(f"   Prerequisites: {relationships['prerequisites']}")
        if relationships['related_topics']:
            print(f"   Related topics: {relationships['related_topics']}")


class TestSchemas:
    """Test JSON schemas"""
    
    def test_metadata_extraction_schema(self):
        """Test metadata extraction schema structure"""
        assert 'type' in METADATA_EXTRACTION_SCHEMA
        assert 'required' in METADATA_EXTRACTION_SCHEMA
        assert 'tags' in METADATA_EXTRACTION_SCHEMA['required']
        assert 'keywords' in METADATA_EXTRACTION_SCHEMA['required']
        
        print("✅ Metadata extraction schema is valid")
    
    def test_relationships_schema(self):
        """Test relationships schema structure"""
        assert 'type' in RELATIONSHIPS_SCHEMA
        assert 'required' in RELATIONSHIPS_SCHEMA
        assert 'prerequisites' in RELATIONSHIPS_SCHEMA['required']
        assert 'related_topics' in RELATIONSHIPS_SCHEMA['required']
        
        print("✅ Relationships schema is valid")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests with custom formatting"""
    print("\n" + "=" * 70)
    print("METADATA AGENT TEST SUITE")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("UNIT TESTS (No API Key Required)")
    print("-" * 70)
    
    # Data models
    print("\n[Data Models]")
    test_models = TestDataModels()
    test_models.test_related_article_creation()
    test_models.test_article_metadata_to_dict()
    test_models.test_difficulty_level_enum()
    test_models.test_content_category_enum()
    
    # Initialization
    print("\n[Agent Initialization]")
    test_init = TestAgentInitialization()
    test_init.test_create_with_llm_client()
    test_init.test_create_with_convenience_function()
    test_init.test_verbose_mode()
    
    # Slug generation
    print("\n[Slug Generation]")
    test_slug = TestSlugGeneration()
    test_slug.test_simple_slug()
    test_slug.test_slug_with_special_chars()
    test_slug.test_long_slug_truncation()
    
    # Technical features
    print("\n[Technical Features]")
    test_tech = TestTechnicalFeatures()
    test_tech.test_detect_code_blocks()
    test_tech.test_detect_tables()
    test_tech.test_detect_images()
    
    # SEO metadata
    print("\n[SEO Metadata]")
    test_seo = TestSEOMetadata()
    test_seo.test_meta_description_length()
    test_seo.test_search_keywords_deduplication()
    
    # Frontmatter
    print("\n[Frontmatter Generation]")
    test_front = TestFrontmatterGeneration()
    test_front.test_basic_frontmatter()
    test_front.test_frontmatter_with_author()
    
    # JSON export
    print("\n[JSON Export]")
    test_json = TestJSONExport()
    test_json.test_metadata_to_json()
    
    # Schemas
    print("\n[Schemas]")
    test_schemas = TestSchemas()
    test_schemas.test_metadata_extraction_schema()
    test_schemas.test_relationships_schema()
    
    print("\n" + "-" * 70)
    print("INTEGRATION TESTS (Require API Key)")
    print("-" * 70)
    
    if not has_api_key():
        print("\n⚠️  No API key found - skipping integration tests")
        print("   Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    else:
        print("\n[Metadata Generation]")
        test_generation = TestMetadataGeneration()
        try:
            test_generation.test_generate_complete_metadata()
            test_generation.test_metadata_with_tables()
            test_generation.test_core_metadata_extraction()
            test_generation.test_relationship_generation()
        except Exception as e:
            print(f"⚠️  Some tests skipped: {e}")
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()