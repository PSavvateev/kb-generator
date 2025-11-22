"""
Agent #1: Analysis Agent for KB Generation Pipeline

This agent analyzes parsed documents and creates structured content plans
that guide the writing agent in creating well-organized KB articles.

Responsibilities:
- Analyze document structure and content
- Identify document type and main topics
- Create section hierarchy
- Determine content elements (lists, tables, code blocks, etc.)
- Plan table placement and formatting
- Generate metadata for content organization

Input: Parsed document (from document_parser.py)
Output: Content plan (JSON structure for writing agent)
"""

import json
import logging
from typing import Dict, List

from services.llm_client import LLMClient
from services.models import ContentPlan, Section, TablePlacement, DocumentType
from config import PipelineConfig

logger = logging.getLogger(__name__)


# ============================================================================
# JSON Schemas for LLM
# ============================================================================

DOCUMENT_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": ["document_type", "main_topic", "target_audience", "key_concepts"],
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [dt.value for dt in DocumentType],
            "description": "Type of document being analyzed"
        },
        "main_topic": {
            "type": "string",
            "description": "Primary topic or subject of the document"
        },
        "target_audience": {
            "type": "string",
            "description": "Intended audience (e.g., 'beginners', 'advanced users', 'developers')"
        },
        "key_concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main concepts covered in the document"
        },
        "estimated_reading_time": {
            "type": "string",
            "description": "Estimated reading time (e.g., '5 minutes', '10-15 minutes')"
        },
        "content_style": {
            "type": "string",
            "description": "Writing style (e.g., 'technical', 'beginner-friendly', 'formal')"
        }
    }
}

SECTION_STRUCTURE_SCHEMA = {
    "type": "object",
    "required": ["sections"],
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "level", "summary", "content_elements"],
                "properties": {
                    "title": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1, "maximum": 4},
                    "summary": {"type": "string"},
                    "content_elements": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "estimated_length": {
                        "type": "string",
                        "enum": ["short", "medium", "long"]
                    }
                }
            }
        }
    }
}

TABLE_PLACEMENT_SCHEMA = {
    "type": "object",
    "required": ["placements"],
    "properties": {
        "placements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["table_index", "section_title", "placement_reason", "should_be_inline"],
                "properties": {
                    "table_index": {"type": "integer"},
                    "section_title": {"type": "string"},
                    "placement_reason": {"type": "string"},
                    "should_be_inline": {"type": "boolean"},
                    "formatting_notes": {"type": "string"}
                }
            }
        }
    }
}


# ============================================================================
# Analysis Agent
# ============================================================================

class AnalysisAgent:
    """
    Agent #1: Analyzes documents and creates content plans
    
    This agent takes parsed document data and generates a structured
    content plan that guides the writing agent in creating KB articles.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: PipelineConfig
    ):
        """
        Initialize Analysis Agent
        
        Args:
            llm_client: LLM client for generating analysis
            config: Pipeline configuration
        """
        self.llm = llm_client
        self.config = config
        
        # Get settings from config
        self.max_retries = config.llm.max_retries
        self.verbose = config.verbose
        
        # Agent-specific settings
        self.extract_key_takeaways = config.agent.analysis_extract_key_takeaways
        self.identify_prerequisites = config.agent.analysis_identify_prerequisites
        self.suggest_related_articles = config.agent.analysis_suggest_related_articles
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized AnalysisAgent with {llm_client.provider.value} provider")


    def analyze(self, parsed_result: Dict) -> ContentPlan:
        """
        Main analysis method - orchestrates the entire analysis process
        
        Args:
            parsed_result: Output from document_parser.py containing:
                - text: Main document text
                - metadata: Document metadata
                - tables: Extracted tables (if any)
                - images: Image information (if any)
        
        Returns:
            ContentPlan: Structured content plan for writing agent
        """
        logger.info("Starting document analysis")
        
        # Validate input
        self._validate_input(parsed_result)
        
        # Step 1: Analyze document characteristics
        doc_analysis = self._analyze_document_type(parsed_result)
        logger.debug(f"Document type identified: {doc_analysis['document_type']}")
        
        # Step 2: Create section structure
        sections = self._create_section_structure(parsed_result, doc_analysis)
        logger.debug(f"Created {len(sections)} main sections")
        
        # Step 3: Plan table placements (if tables exist)
        table_placements = []
        if parsed_result.get('tables'):
            table_placements = self._plan_table_placements(
                parsed_result, 
                sections, 
                doc_analysis
            )
            logger.debug(f"Planned placement for {len(table_placements)} tables")
        
        # Step 4: Generate special instructions
        special_instructions = self._generate_special_instructions(
            parsed_result,
            doc_analysis,
            sections
        )
        
        # Step 5: Assemble complete content plan
        content_plan = ContentPlan(
            document_type=doc_analysis['document_type'],
            main_topic=doc_analysis['main_topic'],
            target_audience=doc_analysis['target_audience'],
            estimated_reading_time=doc_analysis['estimated_reading_time'],
            key_takeaways=doc_analysis['key_concepts'],
            sections=sections,
            table_placements=table_placements,
            content_style=doc_analysis['content_style'],
            special_instructions=special_instructions
        )
        
        logger.info("Document analysis complete")
        return content_plan
    
    def _validate_input(self, parsed_result: Dict):
        """Validate input from parser"""
        required_fields = ['text', 'metadata']
        for field in required_fields:
            if field not in parsed_result:
                raise ValueError(f"Missing required field in parsed_result: {field}")
        
        if not parsed_result['text'].strip():
            raise ValueError("Document text is empty")
    
    def _analyze_document_type(self, parsed_result: Dict) -> Dict:
        """
        Step 1: Analyze document to determine type, audience, and characteristics
        """
        logger.debug("Analyzing document type and characteristics")
        
        text = parsed_result['text']
        metadata = parsed_result.get('metadata', {})
        
        # Build analysis prompt
        system_prompt = """You are an expert document analyst specializing in technical documentation and knowledge base articles.

Your task is to analyze documents and identify their type, target audience, main topics, and writing style.

Consider:
- Document structure and organization
- Language complexity and technical depth
- Presence of instructions, concepts, or reference material
- Intended use case (learning, reference, troubleshooting, etc.)

Be precise and objective in your analysis."""

        user_prompt = f"""Analyze this document and provide a structured analysis.

Document text (first 3000 characters):
{text[:3000]}

Metadata:
{json.dumps(metadata, indent=2)}

Provide your analysis in JSON format with:
- document_type: Type of document (tutorial, how_to, reference, concept, troubleshooting, faq, api_documentation, or general)
- main_topic: Primary topic in one clear phrase
- target_audience: Intended audience (be specific)
- key_concepts: List of 3-7 main concepts covered
- estimated_reading_time: Estimated reading time
- content_style: Writing style description"""

        try:
            analysis = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=DOCUMENT_ANALYSIS_SCHEMA,
                max_retries=self.max_retries
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document type: {e}")
            # Return sensible defaults
            return {
                "document_type": "general",
                "main_topic": "Document analysis",
                "target_audience": "general users",
                "key_concepts": ["main content"],
                "estimated_reading_time": "5-10 minutes",
                "content_style": "informative"
            }
    
    def _create_section_structure(
        self,
        parsed_result: Dict,
        doc_analysis: Dict
    ) -> List[Section]:
        """
        Step 2: Create hierarchical section structure
        """
        logger.debug("Creating section structure")
        
        text = parsed_result['text']
        doc_type = doc_analysis['document_type']
        
        system_prompt = """You are an expert content strategist specializing in creating well-organized knowledge base articles.

Your task is to analyze document content and create a clear, logical section structure that will guide article writing.

Guidelines:
- Create a hierarchical structure with clear section titles
- Each section should have a specific purpose
- Level 1 (H1) is the main title
- Level 2 (H2) are major sections
- Level 3 (H3) are subsections within major sections
- Keep section titles clear, concise, and descriptive
- Identify what types of content elements belong in each section (paragraphs, lists, code, tables, etc.)
- Estimate section length based on content depth

Consider document type and adjust structure accordingly:
- Tutorials: Step-by-step progression
- How-to guides: Clear action-oriented sections
- Reference: Organized by topic/category
- Concepts: Introduction → Explanation → Examples
- Troubleshooting: Problem → Solution format"""

        user_prompt = f"""Create a section structure for this {doc_type} document.

Main topic: {doc_analysis['main_topic']}
Target audience: {doc_analysis['target_audience']}

Document text:
{text[:4000]}

Create a logical section hierarchy with:
- Clear section titles
- Section level (1-4, where 1 is main heading)
- Brief summary of section content
- Content element types (paragraph, bullet_list, numbered_list, code_block, table, etc.)
- Estimated length (short, medium, long)

Provide the structure in JSON format."""

        try:
            structure = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=SECTION_STRUCTURE_SCHEMA,
                max_retries=self.max_retries
            )
            
            # Convert to Section objects
            sections = []
            for section_data in structure.get('sections', []):
                section = Section(
                    title=section_data['title'],
                    level=section_data['level'],
                    summary=section_data['summary'],
                    content_elements=section_data['content_elements'],
                    estimated_length=section_data.get('estimated_length', 'medium'),
                    subsections=None  # Could be extended to support nested sections
                )
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error creating section structure: {e}")
            # Return basic structure
            return [
                Section(
                    title=doc_analysis['main_topic'],
                    level=1,
                    summary="Main content",
                    content_elements=["paragraph"],
                    estimated_length="medium"
                )
            ]
    
    def _plan_table_placements(
        self,
        parsed_result: Dict,
        sections: List[Section],
        doc_analysis: Dict
    ) -> List[TablePlacement]:
        """
        Step 3: Plan where tables should be placed in the article
        """
        logger.debug("Planning table placements")
        
        tables = parsed_result.get('tables', [])
        if not tables:
            return []
        
        # Create section context for the LLM
        section_context = "\n".join([
            f"- {s.title} (Level {s.level}): {s.summary}"
            for s in sections
        ])
        
        # Create table summaries
        table_summaries = []
        for i, table in enumerate(tables):
            # Extract table info
            if isinstance(table, dict):
                rows = table.get('rows', [])
                headers = rows[0] if rows else []
                preview = f"Headers: {headers}, Total rows: {len(rows)}"
            else:
                preview = "Table data available"
            
            table_summaries.append(f"Table {i}: {preview}")
        
        system_prompt = """You are an expert content strategist specializing in document layout and table placement.

Your task is to determine the optimal placement for tables within a knowledge base article.

Guidelines:
- Place tables near the content they support
- Consider table size and whether it should be inline or separate
- Ensure tables enhance understanding rather than interrupt flow
- Provide clear reasoning for placement decisions
- Consider whether tables should be formatted as markdown tables or referenced separately"""

        user_prompt = f"""Plan the placement of tables in this {doc_analysis['document_type']} article.

Article sections:
{section_context}

Tables to place:
{chr(10).join(table_summaries)}

For each table, specify:
- table_index: Table number (0, 1, 2, etc.)
- section_title: Which section it should go in
- placement_reason: Why it belongs in that section
- should_be_inline: Boolean - whether to place inline (true) or reference separately (false)
- formatting_notes: Any special formatting considerations

Provide placements in JSON format."""

        try:
            placements_data = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=TABLE_PLACEMENT_SCHEMA,
                max_retries=self.max_retries
            )
            
            # Convert to TablePlacement objects
            placements = []

            # Handle if placements_data is a list instead of dict
            placement_list = placements_data if isinstance(placements_data, list) else placements_data.get('placements', [])

            for placement_data in placement_list:
                placement = TablePlacement(
                    table_index=placement_data['table_index'],
                    section_title=placement_data['section_title'],
                    placement_reason=placement_data['placement_reason'],
                    should_be_inline=placement_data['should_be_inline'],
                    formatting_notes=placement_data.get('formatting_notes', '')
                )
                placements.append(placement)
            
            return placements
            
        except Exception as e:
            logger.error(f"Error planning table placements: {e}")
            # Return default placements
            return [
                TablePlacement(
                    table_index=i,
                    section_title=sections[0].title if sections else "Content",
                    placement_reason="Default placement",
                    should_be_inline=True,
                    formatting_notes=""
                )
                for i in range(len(tables))
            ]
    
    def _generate_special_instructions(
        self,
        parsed_result: Dict,
        doc_analysis: Dict,
        sections: List[Section]
    ) -> List[str]:
        """
        Step 4: Generate special instructions for the writing agent
        """
        logger.debug("Generating special instructions")
        
        instructions = []
        
        # Document type specific instructions
        doc_type = doc_analysis['document_type']
        
        if doc_type == 'tutorial':
            instructions.append("Write in a step-by-step format with clear progression")
            instructions.append("Include prerequisites section at the beginning")
            instructions.append("Add verification steps after major actions")
        
        elif doc_type == 'how_to':
            instructions.append("Focus on actionable steps and outcomes")
            instructions.append("Use imperative voice (do this, click that)")
            instructions.append("Include tips and warnings where relevant")
        
        elif doc_type == 'reference':
            instructions.append("Maintain consistent formatting for reference entries")
            instructions.append("Include complete parameter/option descriptions")
            instructions.append("Add code examples for technical references")
        
        elif doc_type == 'troubleshooting':
            instructions.append("Use problem-solution format")
            instructions.append("Include symptoms, causes, and solutions")
            instructions.append("Add related issues section")
        
        elif doc_type == 'concept':
            instructions.append("Start with clear definition")
            instructions.append("Progress from simple to complex")
            instructions.append("Include real-world examples")
        
        # Audience specific instructions
        audience = doc_analysis['target_audience'].lower()
        if 'beginner' in audience or 'new' in audience:
            instructions.append("Avoid jargon or explain technical terms")
            instructions.append("Include more examples and explanations")
        
        elif 'advanced' in audience or 'expert' in audience:
            instructions.append("Use technical terminology appropriately")
            instructions.append("Focus on efficiency and best practices")
        
        # Content specific instructions
        if parsed_result.get('tables'):
            instructions.append(f"Document contains {len(parsed_result['tables'])} table(s) - reference them appropriately")
        
        if any('code_block' in s.content_elements for s in sections):
            instructions.append("Include syntax highlighting for code blocks")
            instructions.append("Add comments to complex code examples")
        
        # Style instructions
        content_style = doc_analysis.get('content_style', '').lower()
        if 'formal' in content_style:
            instructions.append("Maintain formal, professional tone")
        elif 'casual' in content_style or 'friendly' in content_style:
            instructions.append("Use conversational, approachable tone")
        
        return instructions
    
    def get_content_plan_json(self, content_plan: ContentPlan) -> str:
        """
        Export content plan as JSON string
        
        Args:
            content_plan: ContentPlan object
            
        Returns:
            JSON string representation
        """
        return json.dumps(content_plan.to_dict(), indent=2)
    
    def save_content_plan(self, content_plan: ContentPlan, filepath: str):
        """
        Save content plan to file
        
        Args:
            content_plan: ContentPlan object
            filepath: Path to save JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.get_content_plan_json(content_plan))
        
        logger.info(f"Content plan saved to {filepath}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_analysis_agent(config: PipelineConfig) -> AnalysisAgent:
    """
    Convenience function to create AnalysisAgent with config
    
    Args:
        config: Pipeline configuration
    
    Returns:
        Configured AnalysisAgent
    """ 
    llm_client = LLMClient(config=config.llm)
    return AnalysisAgent(llm_client, config)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("ANALYSIS AGENT - Example Usage")
    print("=" * 70)
    
    # Example: Create agent
    print("\n1. Creating Analysis Agent...")
    try:
        from ..config import PipelineConfig, LLMProvider, load_env_file
        
        # Load environment
        load_env_file()
        
        # Create config
        config = PipelineConfig()
        config.llm.provider = LLMProvider.GOOGLE  # or OPENAI, ANTHROPIC, OLLAMA
        config.llm.model = "gemini-1.5-flash"
        config.verbose = True
        
        # Create agent
        agent = create_analysis_agent(config)
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        exit(1)
    
    # Example: Sample parsed document
    print("\n2. Creating sample parsed document...")
    sample_parsed_result = {
        "text": """
        Getting Started with Python Flask
        
        Flask is a lightweight web framework for Python that makes it easy to build web applications.
        This guide will walk you through creating your first Flask application.
        
        Prerequisites:
        - Python 3.7 or higher installed
        - Basic knowledge of Python
        - pip package manager
        
        Installation:
        First, install Flask using pip:
        pip install flask
        
        Creating Your First App:
        Create a file called app.py and add the following code:
        
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def hello():
            return 'Hello, World!'
        
        if __name__ == '__main__':
            app.run(debug=True)
        
        Running the Application:
        Run your application with:
        python app.py
        
        Your app will be available at http://localhost:5000
        
        Next Steps:
        - Learn about routing and URL building
        - Explore template rendering with Jinja2
        - Add database support with SQLAlchemy
        """,
        "metadata": {
            "filename": "flask_tutorial.txt",
            "file_type": "text/plain",
            "size": 1234
        },
        "tables": [],
        "images": []
    }
    print("✅ Sample document created")
    
    # Example: Analyze document
    print("\n3. Analyzing document...")
    try:
        content_plan = agent.analyze(sample_parsed_result)
        print("✅ Analysis complete")
        
        # Display results
        print("\n" + "=" * 70)
        print("CONTENT PLAN RESULTS")
        print("=" * 70)
        
        print(f"\n📄 Document Type: {content_plan.document_type}")
        print(f"📌 Main Topic: {content_plan.main_topic}")
        print(f"👥 Target Audience: {content_plan.target_audience}")
        print(f"⏱️  Reading Time: {content_plan.estimated_reading_time}")
        print(f"✍️  Content Style: {content_plan.content_style}")
        
        print(f"\n🎯 Key Takeaways ({len(content_plan.key_takeaways)}):")
        for i, takeaway in enumerate(content_plan.key_takeaways, 1):
            print(f"   {i}. {takeaway}")
        
        print(f"\n📑 Section Structure ({len(content_plan.sections)} sections):")
        for section in content_plan.sections:
            indent = "  " * (section.level - 1)
            print(f"{indent}{'#' * section.level} {section.title}")
            print(f"{indent}   Summary: {section.summary}")
            print(f"{indent}   Elements: {', '.join(section.content_elements)}")
            print(f"{indent}   Length: {section.estimated_length}")
            print()
        
        if content_plan.special_instructions:
            print(f"⚠️  Special Instructions ({len(content_plan.special_instructions)}):")
            for instruction in content_plan.special_instructions:
                print(f"   • {instruction}")
        
        # Save to file
        print("\n4. Saving content plan...")
        agent.save_content_plan(content_plan, "content_plan_example.json")
        print("✅ Saved to: content_plan_example.json")
        
        # Show JSON preview
        print("\n5. JSON Preview (first 500 chars):")
        json_str = agent.get_content_plan_json(content_plan)
        print(json_str[:500] + "...")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)