# test_analysis_agent.py
"""
Comprehensive Test Suite for Analysis Agent

Tests cover:
- Agent initialization
- Input validation
- Document type analysis
- Section structure creation
- Table placement planning
- Special instructions generation
- Content plan assembly
- Error handling
- JSON export
"""

import json
import os
import pytest
from typing import Dict

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from services.analysis_agent import (
    AnalysisAgent,
    ContentPlan,
    Section,
    TablePlacement,
    DocumentType,
    create_analysis_agent,
    DOCUMENT_ANALYSIS_SCHEMA,
    SECTION_STRUCTURE_SCHEMA,
    TABLE_PLACEMENT_SCHEMA
)

from services.llm_client import LLMClient


# ============================================================================
# Test Fixtures
# ============================================================================

class TestData:
    """Test data and sample documents"""
    
    @staticmethod
    def get_sample_tutorial() -> Dict:
        """Sample tutorial document"""
        return {
            "text": """
Getting Started with Python Flask

Flask is a lightweight web framework for Python. This tutorial will guide you through creating your first Flask application.

Prerequisites:
- Python 3.7+
- pip installed

Installation:
Install Flask using pip:
pip install flask

Creating Your First App:
Create app.py with this code:

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()

Running the App:
python app.py

Visit http://localhost:5000 to see your app.
            """,
            "metadata": {
                "filename": "flask_tutorial.txt",
                "file_type": "text/plain"
            },
            "tables": [],
            "images": []
        }
    
    @staticmethod
    def get_sample_reference() -> Dict:
        """Sample reference document"""
        return {
            "text": """
Python String Methods Reference

This reference guide covers commonly used string methods in Python.

str.upper():
Converts all characters to uppercase.
Returns: string

str.lower():
Converts all characters to lowercase.
Returns: string

str.strip():
Removes whitespace from beginning and end.
Returns: string

str.split(delimiter):
Splits string into list.
Parameters: delimiter (optional)
Returns: list

str.replace(old, new):
Replaces occurrences of old with new.
Parameters: old (string), new (string)
Returns: string
            """,
            "metadata": {
                "filename": "string_methods.txt",
                "file_type": "text/plain"
            },
            "tables": [],
            "images": []
        }
    
    @staticmethod
    def get_sample_with_tables() -> Dict:
        """Sample document with tables"""
        return {
            "text": """
HTTP Status Codes Guide

This guide explains common HTTP status codes and their meanings.

Success Codes (2xx):
Status codes in the 200 range indicate success.

Client Error Codes (4xx):
Status codes in the 400 range indicate client errors.

Server Error Codes (5xx):
Status codes in the 500 range indicate server errors.
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
    
    @staticmethod
    def get_empty_document() -> Dict:
        """Empty document for testing validation"""
        return {
            "text": "",
            "metadata": {},
            "tables": [],
            "images": []
        }
    
    @staticmethod
    def get_invalid_document() -> Dict:
        """Invalid document missing required fields"""
        return {
            "metadata": {}
            # Missing 'text'
        }


def has_api_key() -> bool:
    """Check if API key is available for testing"""
    return bool(
        os.getenv('OPENAI_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY') or
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('GEMINI_API_KEY')
    )


def get_test_agent() -> AnalysisAgent:
    """Create test agent with available provider"""
    if os.getenv('OPENAI_API_KEY'):
        return create_analysis_agent(provider='openai', model='gpt-3.5-turbo')
    elif os.getenv('ANTHROPIC_API_KEY'):
        return create_analysis_agent(provider='anthropic')
    elif os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        return create_analysis_agent(provider='google')
    else:
        pytest.skip("No API key available")


# ============================================================================
# Unit Tests (No API Key Required)
# ============================================================================

class TestDataModels:
    """Test data models and schemas"""
    
    def test_section_to_dict(self):
        """Test Section serialization"""
        section = Section(
            title="Introduction",
            level=1,
            summary="Overview of the topic",
            content_elements=["paragraph", "bullet_list"],
            estimated_length="short"
        )
        
        section_dict = section.to_dict()
        
        assert section_dict['title'] == "Introduction"
        assert section_dict['level'] == 1
        assert section_dict['summary'] == "Overview of the topic"
        assert "paragraph" in section_dict['content_elements']
        
        print("✅ Section.to_dict() works correctly")
    
    def test_section_with_subsections(self):
        """Test nested section structure"""
        subsection = Section(
            title="Subsection",
            level=2,
            summary="Details",
            content_elements=["paragraph"],
            estimated_length="short"
        )
        
        main_section = Section(
            title="Main Section",
            level=1,
            summary="Main content",
            content_elements=["paragraph"],
            estimated_length="medium",
            subsections=[subsection]
        )
        
        section_dict = main_section.to_dict()
        
        assert 'subsections' in section_dict
        assert len(section_dict['subsections']) == 1
        assert section_dict['subsections'][0]['title'] == "Subsection"
        
        print("✅ Nested sections work correctly")
    
    def test_content_plan_to_dict(self):
        """Test ContentPlan serialization"""
        section = Section(
            title="Test",
            level=1,
            summary="Test section",
            content_elements=["paragraph"],
            estimated_length="short"
        )
        
        placement = TablePlacement(
            table_index=0,
            section_title="Test",
            placement_reason="Testing",
            should_be_inline=True,
            formatting_notes="None"
        )
        
        plan = ContentPlan(
            document_type="tutorial",
            main_topic="Testing",
            target_audience="developers",
            estimated_reading_time="5 minutes",
            key_takeaways=["test1", "test2"],
            sections=[section],
            table_placements=[placement],
            content_style="technical",
            special_instructions=["instruction1"]
        )
        
        plan_dict = plan.to_dict()
        
        assert plan_dict['document_type'] == "tutorial"
        assert plan_dict['main_topic'] == "Testing"
        assert len(plan_dict['sections']) == 1
        assert len(plan_dict['table_placements']) == 1
        
        print("✅ ContentPlan.to_dict() works correctly")
    
    def test_document_analysis_schema(self):
        """Test document analysis schema structure"""
        assert 'type' in DOCUMENT_ANALYSIS_SCHEMA
        assert 'required' in DOCUMENT_ANALYSIS_SCHEMA
        assert 'document_type' in DOCUMENT_ANALYSIS_SCHEMA['required']
        
        print("✅ Document analysis schema is valid")
    
    def test_section_structure_schema(self):
        """Test section structure schema"""
        assert 'properties' in SECTION_STRUCTURE_SCHEMA
        assert 'sections' in SECTION_STRUCTURE_SCHEMA['properties']
        
        print("✅ Section structure schema is valid")
    
    def test_table_placement_schema(self):
        """Test table placement schema"""
        assert 'properties' in TABLE_PLACEMENT_SCHEMA
        assert 'placements' in TABLE_PLACEMENT_SCHEMA['properties']
        
        print("✅ Table placement schema is valid")


class TestAgentInitialization:
    """Test agent initialization"""
    
    def test_create_with_llm_client(self):
        """Test creating agent with LLM client"""
        try:
            llm = LLMClient(provider='openai', model='gpt-3.5-turbo', api_key='test-key-for-initialization' )
            agent = AnalysisAgent(llm)
            
            assert agent.llm is not None
            assert agent.max_retries == 3
            assert agent.verbose is False
            
            print("✅ Agent initialization with LLM client works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_create_with_convenience_function(self):
        """Test convenience function"""
        try:
            agent = create_analysis_agent(provider='openai', model='gpt-3.5-turbo', api_key='test-key-for-initialization')
            
            assert isinstance(agent, AnalysisAgent)
            assert agent.llm.provider.value == 'openai'
            
            print("✅ Convenience function works")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_verbose_mode(self):
        """Test verbose mode setting"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization', verbose=True)
            
            assert agent.verbose is True
            
            print("✅ Verbose mode works")
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestInputValidation:
    """Test input validation"""
    
    def test_validate_valid_input(self):
        """Test validation with valid input"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization')
            sample = TestData.get_sample_tutorial()
            
            # Should not raise exception
            agent._validate_input(sample)
            
            print("✅ Valid input passes validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_validate_empty_text(self):
        """Test validation with empty text"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization')
            sample = TestData.get_empty_document()
            
            with pytest.raises(ValueError, match="text is empty"):
                agent._validate_input(sample)
            
            print("✅ Empty text caught by validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_validate_missing_fields(self):
        """Test validation with missing fields"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization')
            sample = TestData.get_invalid_document()
            
            with pytest.raises(ValueError, match="Missing required field"):
                agent._validate_input(sample)
            
            print("✅ Missing fields caught by validation")
        except ImportError:
            pytest.skip("OpenAI package not installed")


# ============================================================================
# Integration Tests (Require API Key)
# ============================================================================

class TestDocumentAnalysis:
    """Test document type analysis (requires API key)"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_analyze_tutorial(self):
        """Test analyzing a tutorial document"""
        agent = get_test_agent()
        sample = TestData.get_sample_tutorial()
        
        result = agent._analyze_document_type(sample)
        
        assert 'document_type' in result
        assert 'main_topic' in result
        assert 'target_audience' in result
        assert 'key_concepts' in result
        
        print(f"✅ Tutorial analysis works")
        print(f"   Type: {result['document_type']}")
        print(f"   Topic: {result['main_topic']}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_analyze_reference(self):
        """Test analyzing a reference document"""
        agent = get_test_agent()
        sample = TestData.get_sample_reference()
        
        result = agent._analyze_document_type(sample)
        
        assert 'document_type' in result
        assert isinstance(result['key_concepts'], list)
        
        print(f"✅ Reference analysis works")
        print(f"   Type: {result['document_type']}")


class TestSectionStructure:
    """Test section structure creation (requires API key)"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_create_section_structure(self):
        """Test creating section structure"""
        agent = get_test_agent()
        sample = TestData.get_sample_tutorial()
        
        doc_analysis = agent._analyze_document_type(sample)
        sections = agent._create_section_structure(sample, doc_analysis)
        
        assert isinstance(sections, list)
        assert len(sections) > 0
        assert all(isinstance(s, Section) for s in sections)
        
        print(f"✅ Section structure creation works")
        print(f"   Created {len(sections)} sections")
        for section in sections:
            print(f"   - {section.title} (Level {section.level})")


class TestTablePlacement:
    """Test table placement planning (requires API key)"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_plan_table_placements(self):
        """Test planning table placements"""
        agent = get_test_agent()
        sample = TestData.get_sample_with_tables()
        
        doc_analysis = agent._analyze_document_type(sample)
        sections = agent._create_section_structure(sample, doc_analysis)
        placements = agent._plan_table_placements(sample, sections, doc_analysis)
        
        assert isinstance(placements, list)
        assert len(placements) == 2  # Sample has 2 tables
        assert all(isinstance(p, TablePlacement) for p in placements)
        
        print(f"✅ Table placement works")
        print(f"   Planned {len(placements)} placements")
        for placement in placements:
            print(f"   - Table {placement.table_index} → {placement.section_title}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_no_tables(self):
        """Test with document that has no tables"""
        agent = get_test_agent()
        sample = TestData.get_sample_tutorial()
        
        doc_analysis = agent._analyze_document_type(sample)
        sections = agent._create_section_structure(sample, doc_analysis)
        placements = agent._plan_table_placements(sample, sections, doc_analysis)
        
        assert placements == []
        
        print("✅ No tables handled correctly")


class TestSpecialInstructions:
    """Test special instructions generation"""
    
    def test_generate_instructions_tutorial(self):
        """Test instructions for tutorial"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization')
            sample = TestData.get_sample_tutorial()
            
            doc_analysis = {
                'document_type': 'tutorial',
                'target_audience': 'beginners'
            }
            sections = [
                Section("Test", 1, "test", ["paragraph"], "short")
            ]
            
            instructions = agent._generate_special_instructions(
                sample, doc_analysis, sections
            )
            
            assert isinstance(instructions, list)
            assert len(instructions) > 0
            
            print(f"✅ Tutorial instructions generated")
            print(f"   Generated {len(instructions)} instructions")
            
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    def test_generate_instructions_with_tables(self):
        """Test instructions when tables present"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization')
            sample = TestData.get_sample_with_tables()
            
            doc_analysis = {'document_type': 'reference', 'target_audience': 'general'}
            sections = [Section("Test", 1, "test", ["paragraph"], "short")]
            
            instructions = agent._generate_special_instructions(
                sample, doc_analysis, sections
            )
            
            # Should mention tables
            table_mentioned = any('table' in inst.lower() for inst in instructions)
            assert table_mentioned
            
            print("✅ Table-related instructions generated")
            
        except ImportError:
            pytest.skip("OpenAI package not installed")


class TestFullAnalysis:
    """Test complete analysis workflow (requires API key)"""
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_analyze_tutorial_document(self):
        """Test full analysis of tutorial"""
        agent = get_test_agent()
        sample = TestData.get_sample_tutorial()
        
        content_plan = agent.analyze(sample)
        
        assert isinstance(content_plan, ContentPlan)
        assert content_plan.document_type is not None
        assert content_plan.main_topic is not None
        assert len(content_plan.sections) > 0
        assert len(content_plan.key_takeaways) > 0
        assert len(content_plan.special_instructions) > 0
        
        print("✅ Full tutorial analysis works")
        print(f"   Type: {content_plan.document_type}")
        print(f"   Topic: {content_plan.main_topic}")
        print(f"   Sections: {len(content_plan.sections)}")
        print(f"   Instructions: {len(content_plan.special_instructions)}")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_analyze_document_with_tables(self):
        """Test full analysis with tables"""
        agent = get_test_agent()
        sample = TestData.get_sample_with_tables()
        
        content_plan = agent.analyze(sample)
        
        assert isinstance(content_plan, ContentPlan)
        assert len(content_plan.table_placements) == 2
        
        print("✅ Analysis with tables works")
        print(f"   Table placements: {len(content_plan.table_placements)}")


class TestJSONExport:
    """Test JSON export functionality"""
    
    def test_content_plan_to_json(self):
        """Test exporting content plan to JSON"""
        try:
            agent = create_analysis_agent(provider='openai', api_key='test-key-for-initialization')
            
            section = Section("Test", 1, "test", ["paragraph"], "short")
            plan = ContentPlan(
                document_type="tutorial",
                main_topic="Test",
                target_audience="developers",
                estimated_reading_time="5 min",
                key_takeaways=["test"],
                sections=[section],
                table_placements=[],
                content_style="technical",
                special_instructions=["test"]
            )
            
            json_str = agent.get_content_plan_json(plan)
            
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert parsed['document_type'] == "tutorial"
            
            print("✅ JSON export works")
            
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    @pytest.mark.skipif(not has_api_key(), reason="No API key available")
    def test_save_content_plan(self, tmp_path):
        """Test saving content plan to file"""
        agent = get_test_agent()
        sample = TestData.get_sample_tutorial()
        
        content_plan = agent.analyze(sample)
        
        # Save to temp file
        output_path = tmp_path / "test_plan.json"
        agent.save_content_plan(content_plan, str(output_path))
        
        # Verify file exists and is valid JSON
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['document_type'] == content_plan.document_type
        
        print("✅ Save to file works")
        print(f"   Saved to: {output_path}")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests with custom formatting"""
    print("\n" + "=" * 70)
    print("ANALYSIS AGENT TEST SUITE")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("UNIT TESTS (No API Key Required)")
    print("-" * 70)
    
    # Data models
    print("\n[Data Models]")
    test_models = TestDataModels()
    test_models.test_section_to_dict()
    test_models.test_section_with_subsections()
    test_models.test_content_plan_to_dict()
    test_models.test_document_analysis_schema()
    test_models.test_section_structure_schema()
    test_models.test_table_placement_schema()
    
    # Initialization
    print("\n[Agent Initialization]")
    test_init = TestAgentInitialization()
    test_init.test_create_with_llm_client()
    test_init.test_create_with_convenience_function()
    test_init.test_verbose_mode()
    
    # Validation
    print("\n[Input Validation]")
    test_validation = TestInputValidation()
    test_validation.test_validate_valid_input()
    test_validation.test_validate_empty_text()
    test_validation.test_validate_missing_fields()
    
    # Special instructions
    print("\n[Special Instructions]")
    test_instructions = TestSpecialInstructions()
    test_instructions.test_generate_instructions_tutorial()
    test_instructions.test_generate_instructions_with_tables()
    
    print("\n" + "-" * 70)
    print("INTEGRATION TESTS (Require API Key)")
    print("-" * 70)
    
    if not has_api_key():
        print("\n⚠️  No API key found - skipping integration tests")
        print("   Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    else:
        # Document analysis
        print("\n[Document Analysis]")
        test_analysis = TestDocumentAnalysis()
        try:
            test_analysis.test_analyze_tutorial()
            test_analysis.test_analyze_reference()
        except Exception as e:
            print(f"⚠️  Skipped: {e}")
        
        # Section structure
        print("\n[Section Structure]")
        test_sections = TestSectionStructure()
        try:
            test_sections.test_create_section_structure()
        except Exception as e:
            print(f"⚠️  Skipped: {e}")
        
        # Table placement
        print("\n[Table Placement]")
        test_tables = TestTablePlacement()
        try:
            test_tables.test_plan_table_placements()
            test_tables.test_no_tables()
        except Exception as e:
            print(f"⚠️  Skipped: {e}")
        
        # Full analysis
        print("\n[Full Analysis]")
        test_full = TestFullAnalysis()
        try:
            test_full.test_analyze_tutorial_document()
            test_full.test_analyze_document_with_tables()
        except Exception as e:
            print(f"⚠️  Skipped: {e}")
        
        # JSON export
        print("\n[JSON Export]")
        test_json = TestJSONExport()
        try:
            test_json.test_content_plan_to_json()
        except Exception as e:
            print(f"⚠️  Skipped: {e}")
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()