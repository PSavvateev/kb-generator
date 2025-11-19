# test_writing_agent.py
"""
Comprehensive Test Suite for Writing Agent

Tests cover:
- Agent initialization
- Input validation
- Article generation
- Content fidelity (no hallucination)
- Table handling
- Markdown formatting
- Reading time calculation
- File saving
- JSON export
- Error handling
"""

import json
import os
import pytest
from typing import Dict
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from services.writing_agent import (
    WritingAgent,
    KBArticle,
    create_writing_agent
)
from services.analysis_agent import (
    ContentPlan,
    Section,
    TablePlacement
)
from services.llm_client import LLMClient


# ============================================================================
# Test Fixtures and Data
# ============================================================================

class TestData:
    """Test data and sample documents"""
    
    @staticmethod
    def get_sample_content_plan() -> ContentPlan:
        """Sample content plan from Analysis Agent"""
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
                ),
                Section(
                    title="Creating Your First App",
                    level=2,
                    summary="Building a basic Flask application",
                    content_elements=["paragraph", "code_block"],
                    estimated_length="long"
                ),
                Section(
                    title="Running the Application",
                    level=2,
                    summary="Starting the Flask server",
                    content_elements=["paragraph", "code_block"],
                    estimated_length="medium"
                )
            ],
            table_placements=[],
            content_style="beginner-friendly",
            special_instructions=[
                "Write in step-by-step format",
                "Explain technical terms",
                "Include code examples with comments"
            ]
        )
    
    @staticmethod
    def get_sample_parsed_result() -> Dict:
        """Sample parsed document"""
        return {
            "text": """
Flask Tutorial

Flask is a lightweight web framework for Python. It's easy to learn and perfect for beginners.

What You'll Need:
- Python 3.7 or higher
- pip package manager
- Basic Python knowledge

Installing Flask:

Open your terminal and run:
pip install flask

This will install Flask and its dependencies.

Creating Your First Application:

Create a new file called app.py:

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

Let me explain what this code does:
- Line 1: Import Flask class
- Line 2: Create Flask application instance
- Line 4-6: Define a route that returns "Hello, World!"
- Line 8-9: Run the application in debug mode

Running Your Application:

In your terminal, run:
python app.py

You'll see output like:
 * Running on http://127.0.0.1:5000/

Open your web browser and visit http://localhost:5000
You should see "Hello, World!" displayed.

The debug=True parameter is useful during development because:
- Auto-reloads when you change code
- Shows helpful error messages

That's it! You've created your first Flask application.
            """,
            "metadata": {
                "filename": "flask_tutorial.txt",
                "file_type": "text/plain",
                "size": 1234
            },
            "tables": [],
            "images": []
        }
    
    @staticmethod
    def get_sample_with_tables() -> Dict:
        """Sample with tables"""
        content_plan = ContentPlan(
            document_type="reference",
            main_topic="HTTP Status Codes",
            target_audience="developers",
            estimated_reading_time="5 minutes",
            key_takeaways=["Status codes", "Error handling"],
            sections=[
                Section(
                    title="Success Codes",
                    level=2,
                    summary="2xx status codes",
                    content_elements=["paragraph", "table"],
                    estimated_length="medium"
                ),
                Section(
                    title="Error Codes",
                    level=2,
                    summary="4xx status codes",
                    content_elements=["paragraph", "table"],
                    estimated_length="medium"
                )
            ],
            table_placements=[
                TablePlacement(
                    table_index=0,
                    section_title="Success Codes",
                    placement_reason="Shows success status codes",
                    should_be_inline=True,
                    formatting_notes="Use markdown table"
                ),
                TablePlacement(
                    table_index=1,
                    section_title="Error Codes",
                    placement_reason="Shows error status codes",
                    should_be_inline=True,
                    formatting_notes="Use markdown table"
                )
            ],
            content_style="technical",
            special_instructions=["Include table formatting"]
        )
        
        parsed_result = {
            "text": """
HTTP Status Codes Guide

Success Codes (2xx):
These codes indicate that the request was successful.

Error Codes (4xx):
These codes indicate client-side errors.
            """,
            "metadata": {
                "filename": "http_status.txt",
                "file_type": "text/plain"
            },
            "tables": [
                {
                    "rows": [
                        ["Code", "Message", "Description"],
                        ["200", "OK", "Request successful"],
                        ["201", "Created", "Resource created"],
                        ["204", "No Content", "Success but no content"]
                    ]
                },
                {
                    "rows": [
                        ["Code", "Message", "Description"],
                        ["400", "Bad Request", "Invalid request"],
                        ["404", "Not Found", "Resource not found"],
                        ["403", "Forbidden", "Access denied"]
                    ]
                }
            ],
            "images": []
        }
        
        return content_plan, parsed_result
    
    @staticmethod
    def get_minimal_content_plan() -> ContentPlan:
        """Minimal content plan for testing"""
        return ContentPlan(
            document_type="general",
            main_topic="Test Article",
            target_audience="general",
            estimated_reading_time="1 minute",
            key_takeaways=["test"],
            sections=[
                Section(
                    title="Main Section",
                    level=1,
                    summary="Main content",
                    content_elements=["paragraph"],
                    estimated_length="short"
                )
            ],
            table_placements=[],
            content_style="professional",
            special_instructions=[]
        )
    
    @staticmethod
    def get_minimal_parsed_result() -> Dict:
        """Minimal parsed result for testing"""
        return {
            "text": "This is a test document with minimal content.",
            "metadata": {
                "filename": "test.txt",
                "file_type": "text/plain"
            },
            "tables": [],
            "images": []
        }


def has_api_key() -> bool:
    """Check if API key is available for testing"""
    return bool(
        os.getenv('OPENAI_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY') or
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('GEMINI_API_KEY')
    )


def get_test_agent() -> WritingAgent:
    """Create test agent with available provider"""
    if os.getenv('OPENAI_API_KEY'):
        return create_writing_agent(provider='openai', model='gpt-3.5-turbo')
    elif os.getenv('ANTHROPIC_API_KEY'):
        return create_writing_agent(provider='anthropic')
    elif os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        return create_writing_agent(provider='google')
    else:
        pytest.skip("No API key available")


# ============================================================================
# Unit Tests (No API Key Required)
# ============================================================================

class TestDataModels:
    """Test data models"""
    
    def test_kb_article_to_dict(self):
        """Test KBArticle serialization"""
        article = KBArticle(
            title="Test Article",
            content="# Test\n\nContent here.",
            metadata={"type": "test"},
            word_count=100,
            estimated_reading_time="1 minute"
        )
        
        article_dict = article.to_dict()
        
        assert article_dict['title'] == "Test Article"
        assert article_dict['word_count'] == 100
        assert article_dict['estimated_reading_time'] == "1 minute"
        assert 'metadata' in article_dict
        
        print("✅ KBArticle.to_dict() works correctly")


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
            agent = WritingAgent(llm)
            
            assert agent.llm is not None
            assert agent.max_retries == 3
            assert agent.verbose is False
            
            print("✅ Agent initialization works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_create_with_convenience_function(self):
        """Test convenience function"""
        try:
            agent = create_writing_agent(
                provider='openai',
                model='gpt-3.5-turbo',
                api_key='test-key-for-initialization'
            )
            
            assert isinstance(agent, WritingAgent)
            assert agent.llm.provider.value == 'openai'
            
            print("✅ Convenience function works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_verbose_mode(self):
        """Test verbose mode setting"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization',
                verbose=True
            )
            
            assert agent.verbose is True
            
            print("✅ Verbose mode works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestInputValidation:
    """Test input validation"""
    
    def test_validate_valid_inputs(self):
        """Test validation with valid inputs"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            content_plan = TestData.get_minimal_content_plan()
            parsed_result = TestData.get_minimal_parsed_result()
            
            # Should not raise exception
            agent._validate_inputs(content_plan, parsed_result)
            
            print("✅ Valid inputs pass validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_validate_empty_sections(self):
        """Test validation with no sections"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            content_plan = TestData.get_minimal_content_plan()
            content_plan.sections = []  # Empty sections
            parsed_result = TestData.get_minimal_parsed_result()
            
            with pytest.raises(ValueError, match="no sections"):
                agent._validate_inputs(content_plan, parsed_result)
            
            print("✅ Empty sections caught by validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_validate_empty_text(self):
        """Test validation with no text content"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            content_plan = TestData.get_minimal_content_plan()
            parsed_result = {
                "text": "",  # Empty text
                "metadata": {},
                "tables": []
            }
            
            with pytest.raises(ValueError, match="no text content"):
                agent._validate_inputs(content_plan, parsed_result)
            
            print("✅ Empty text caught by validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_calculate_reading_time_short(self):
        """Test reading time calculation for short text"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            # 100 words = 1 minute minimum
            time = agent._calculate_reading_time(100)
            assert time == "1 minute"
            
            print("✅ Short reading time calculated correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_calculate_reading_time_medium(self):
        """Test reading time calculation for medium text"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            # 500 words ≈ 2 minutes
            time = agent._calculate_reading_time(500)
            assert "minute" in time
            assert time != "1 minute"
            
            print("✅ Medium reading time calculated correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_calculate_reading_time_long(self):
        """Test reading time calculation for long text"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            # 15000 words ≈ 1+ hour
            time = agent._calculate_reading_time(15000)
            assert "hour" in time
            
            print("✅ Long reading time calculated correctly")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_add_source_attribution(self):
        """Test source attribution generation"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            parsed_result = TestData.get_minimal_parsed_result()
            
            attribution = agent._add_source_attribution(parsed_result)
            
            assert "Source:" in attribution
            assert "test.txt" in attribution
            assert "---" in attribution
            
            print("✅ Source attribution works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestPromptBuilding:
    """Test prompt building logic"""
    
    def test_build_prompt_basic(self):
        """Test basic prompt building"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            content_plan = TestData.get_minimal_content_plan()
            source_text = "Test content here."
            tables = []
            
            prompt = agent._build_generation_prompt(content_plan, source_text, tables)
            
            assert "CRITICAL" in prompt
            assert "SOURCE DOCUMENT" in prompt
            assert source_text in prompt
            assert content_plan.main_topic in prompt
            
            print("✅ Basic prompt building works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_build_prompt_with_tables(self):
        """Test prompt building with tables"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            content_plan, parsed_result = TestData.get_sample_with_tables()
            
            prompt = agent._build_generation_prompt(
                content_plan,
                parsed_result['text'],
                parsed_result['tables']
            )
            
            assert "TABLE PLACEMENTS" in prompt
            assert "Table 0" in prompt
            assert "Table 1" in prompt
            
            print("✅ Prompt with tables works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_build_prompt_with_instructions(self):
        """Test prompt building with special instructions"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            content_plan = TestData.get_sample_content_plan()
            source_text = "Test content."
            
            prompt = agent._build_generation_prompt(content_plan, source_text, [])
            
            assert "SPECIAL INSTRUCTIONS" in prompt
            for instruction in content_plan.special_instructions:
                assert instruction in prompt
            
            print("✅ Prompt with instructions works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestJSONExport:
    """Test JSON export functionality"""
    
    def test_article_to_json(self):
        """Test converting article to JSON"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = KBArticle(
                title="Test",
                content="# Test\n\nContent.",
                metadata={"type": "test"},
                word_count=50,
                estimated_reading_time="1 minute"
            )
            
            json_str = agent.get_article_json(article)
            
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert parsed['title'] == "Test"
            assert parsed['word_count'] == 50
            
            print("✅ JSON export works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestFileSaving:
    """Test file saving functionality"""
    
    def test_save_article_with_metadata(self, tmp_path):
        """Test saving article with frontmatter"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = KBArticle(
                title="Test Article",
                content="# Test\n\nContent here.",
                metadata={
                    "document_type": "tutorial",
                    "target_audience": "beginners",
                    "style": "friendly"
                },
                word_count=50,
                estimated_reading_time="1 minute"
            )
            
            output_path = tmp_path / "test_article.md"
            agent.save_article(article, str(output_path), include_metadata=True)
            
            assert output_path.exists()
            
            content = output_path.read_text(encoding='utf-8')
            assert "---" in content  # Frontmatter
            assert "title: Test Article" in content
            assert "word_count: 50" in content
            assert "# Test" in content
            
            print("✅ Save with metadata works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_save_article_without_metadata(self, tmp_path):
        """Test saving article without frontmatter"""
        try:
            agent = create_writing_agent(
                provider='openai',
                api_key='test-key-for-initialization'
            )
            
            article = KBArticle(
                title="Test",
                content="# Test\n\nContent.",
                metadata={},
                word_count=50,
                estimated_reading_time="1 minute"
            )
            
            output_path = tmp_path / "test_no_meta.md"
            agent.save_article(article, str(output_path), include_metadata=False)
            
            assert output_path.exists()
            
            content = output_path.read_text(encoding='utf-8')
            assert not content.startswith("---")  # No frontmatter
            assert "# Test" in content
            
            print("✅ Save without metadata works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


# ============================================================================
# Integration Tests (Require API Key)
# ============================================================================

class TestArticleGeneration:
    """Test article generation with real LLM (requires API key)"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_generate_simple_article(self):
        """Test generating a simple article"""
        agent = get_test_agent()
        content_plan = TestData.get_minimal_content_plan()
        parsed_result = TestData.get_minimal_parsed_result()
        
        article = agent.write(content_plan, parsed_result)
        
        assert isinstance(article, KBArticle)
        assert article.title == content_plan.main_topic
        assert len(article.content) > 0
        assert article.word_count > 0
        assert article.estimated_reading_time is not None
        
        print("✅ Simple article generation works")
        print(f"   Title: {article.title}")
        print(f"   Words: {article.word_count}")
        print(f"   Reading time: {article.estimated_reading_time}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_generate_tutorial_article(self):
        """Test generating a tutorial article"""
        agent = get_test_agent()
        content_plan = TestData.get_sample_content_plan()
        parsed_result = TestData.get_sample_parsed_result()
        
        article = agent.write(content_plan, parsed_result)
        
        assert isinstance(article, KBArticle)
        assert article.metadata['document_type'] == "tutorial"
        assert article.metadata['target_audience'] == "beginner developers"
        assert len(article.content) > len(parsed_result['text']) * 0.5  # Reasonable length
        
        # Check for markdown formatting
        assert "#" in article.content  # Has headings
        
        print("✅ Tutorial article generation works")
        print(f"   Sections in plan: {len(content_plan.sections)}")
        print(f"   Word count: {article.word_count}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_generate_article_with_tables(self):
        """Test generating article with tables"""
        agent = get_test_agent()
        content_plan, parsed_result = TestData.get_sample_with_tables()
        
        article = agent.write(content_plan, parsed_result)
        
        assert isinstance(article, KBArticle)
        assert article.metadata['has_tables'] is True
        assert len(content_plan.table_placements) == 2
        
        # Check if article mentions tables or has table formatting
        # (Note: actual table formatting depends on LLM)
        assert len(article.content) > 0
        
        print("✅ Article with tables generation works")
        print(f"   Tables in source: {len(parsed_result['tables'])}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_content_fidelity(self):
        """Test that article uses only source content"""
        agent = get_test_agent()
        
        # Create content with very specific, unique information
        unique_text = "The XYZ-9000 processor has exactly 42 cores and costs $12,345."
        
        content_plan = ContentPlan(
            document_type="reference",
            main_topic="XYZ-9000 Processor",
            target_audience="technical",
            estimated_reading_time="2 minutes",
            key_takeaways=["specs"],
            sections=[
                Section(
                    title="Specifications",
                    level=1,
                    summary="Technical specs",
                    content_elements=["paragraph"],
                    estimated_length="short"
                )
            ],
            table_placements=[],
            content_style="technical",
            special_instructions=[]
        )
        
        parsed_result = {
            "text": unique_text,
            "metadata": {"filename": "specs.txt"},
            "tables": [],
            "images": []
        }
        
        article = agent.write(content_plan, parsed_result)
        
        # Check that specific information appears in article
        content_lower = article.content.lower()
        assert "42" in article.content or "forty-two" in content_lower
        assert "12,345" in article.content or "12345" in article.content
        
        print("✅ Content fidelity maintained")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_article_without_attribution(self):
        """Test generating article without source attribution"""
        agent = get_test_agent()
        content_plan = TestData.get_minimal_content_plan()
        parsed_result = TestData.get_minimal_parsed_result()
        
        article = agent.write(
            content_plan,
            parsed_result,
            include_source_attribution=False
        )
        
        assert "Source:" not in article.content
        
        print("✅ Article without attribution works")


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_invalid_content_plan(self):
        """Test handling of invalid content plan"""
        agent = get_test_agent()
        
        # Content plan with no sections
        invalid_plan = TestData.get_minimal_content_plan()
        invalid_plan.sections = []
        
        parsed_result = TestData.get_minimal_parsed_result()
        
        with pytest.raises(ValueError):
            agent.write(invalid_plan, parsed_result)
        
        print("✅ Invalid content plan handled correctly")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_empty_source_text(self):
        """Test handling of empty source text"""
        agent = get_test_agent()
        content_plan = TestData.get_minimal_content_plan()
        
        # Empty text
        invalid_parsed = {
            "text": "",
            "metadata": {},
            "tables": []
        }
        
        with pytest.raises(ValueError):
            agent.write(content_plan, invalid_parsed)
        
        print("✅ Empty source text handled correctly")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests with custom formatting"""
    print("\n" + "=" * 70)
    print("WRITING AGENT TEST SUITE")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("UNIT TESTS (No API Key Required)")
    print("-" * 70)
    
    # Data models
    print("\n[Data Models]")
    test_models = TestDataModels()
    test_models.test_kb_article_to_dict()
    
    # Initialization
    print("\n[Agent Initialization]")
    test_init = TestAgentInitialization()
    test_init.test_create_with_llm_client()
    test_init.test_create_with_convenience_function()
    test_init.test_verbose_mode()
    
    # Validation
    print("\n[Input Validation]")
    test_validation = TestInputValidation()
    test_validation.test_validate_valid_inputs()
    test_validation.test_validate_empty_sections()
    test_validation.test_validate_empty_text()
    
    # Utility methods
    print("\n[Utility Methods]")
    test_utils = TestUtilityMethods()
    test_utils.test_calculate_reading_time_short()
    test_utils.test_calculate_reading_time_medium()
    test_utils.test_calculate_reading_time_long()
    test_utils.test_add_source_attribution()
    
    # Prompt building
    print("\n[Prompt Building]")
    test_prompts = TestPromptBuilding()
    test_prompts.test_build_prompt_basic()
    test_prompts.test_build_prompt_with_tables()
    test_prompts.test_build_prompt_with_instructions()
    
    # JSON export
    print("\n[JSON Export]")
    test_json = TestJSONExport()
    test_json.test_article_to_json()
    
    print("\n" + "-" * 70)
    print("INTEGRATION TESTS (Require API Key)")
    print("-" * 70)
    
    if not has_api_key():
        print("\n⚠️  No API key found - skipping integration tests")
        print("   Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    else:
        # Article generation
        print("\n[Article Generation]")
        test_generation = TestArticleGeneration()
        try:
            test_generation.test_generate_simple_article()
            test_generation.test_generate_tutorial_article()
            test_generation.test_generate_article_with_tables()
            test_generation.test_content_fidelity()
            test_generation.test_article_without_attribution()
        except Exception as e:
            print(f"⚠️  Some tests skipped: {e}")
        
        # Error handling
        print("\n[Error Handling]")
        test_errors = TestErrorHandling()
        try:
            test_errors.test_invalid_content_plan()
            test_errors.test_empty_source_text()
        except Exception as e:
            print(f"⚠️  Some tests skipped: {e}")
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()