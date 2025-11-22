# writing_agent.py
"""
Agent #2: Writing Agent for KB Generation Pipeline

This agent generates well-structured KB articles based on content plans
from the Analysis Agent and source material from the Document Parser.

CRITICAL: This agent ONLY uses information from the source document.
It does NOT add information from the LLM's training data.
It restructures and reformats existing content ONLY.

Responsibilities:
- Transform parsed content into structured KB article
- Follow content plan structure from Analysis Agent
- Maintain source material fidelity (NO hallucination)
- Format content according to KB standards
- Place tables and code blocks appropriately
- Generate clear, well-organized markdown

Input: 
- Content plan (from analysis_agent.py)
- Parsed document (from document_parser.py)

Output: 
- Formatted KB article (Markdown)
"""

import json
import logging
from typing import Dict, List, Optional

from services.llm_client import LLMClient
from services.models import ContentPlan, KBArticle
from config import PipelineConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Writing Agent
# ============================================================================

class WritingAgent:
    """
    Agent #2: Transforms content into structured KB articles
    
    CRITICAL RULES:
    1. ONLY use information from the source document
    2. DO NOT add external information or examples
    3. DO NOT make up facts, statistics, or details
    4. Restructure and reformat existing content ONLY
    5. If information is missing, leave it out
    """
    
    # Strict system prompt emphasizing content fidelity
    SYSTEM_PROMPT = """ You are a technical writer creating knowledge base articles.

                    CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:

                    1. CONTENT FIDELITY: 
                    - Use ONLY information from the provided source document
                    - DO NOT add information from your training data
                    - DO NOT create examples unless they exist in the source
                    - DO NOT add explanations beyond what's in the source
                    - If something isn't in the source, DON'T include it

                    2. YOUR ROLE:
                    - Restructure existing content into clear sections
                    - Improve clarity and organization
                    - Fix grammar and formatting
                    - Maintain all technical accuracy from source
                    - Use professional KB article style

                    3. WHAT YOU CAN DO:
                    - Reorganize content for better flow
                    - Improve sentence structure
                    - Add appropriate headings
                    - Format code blocks and lists
                    - Create clear transitions

                    4. WHAT YOU CANNOT DO:
                    - Add information not in the source
                    - Create fictional examples
                    - Add your own knowledge
                    - Make assumptions about missing information
                    - Embellish or expand beyond the source

                    5. IF INFORMATION IS MISSING:
                    - Leave it out
                    - Don't try to fill gaps
                    - Work with what's provided

                    Your output must be a faithful representation of the source material,
                    just better organized and formatted.
                """
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: PipelineConfig
    ):
        """
        Initialize Writing Agent
        
        Args:
            llm_client: LLM client for generating content
            config: Pipeline configuration
        """
        self.llm = llm_client
        self.config = config
        
        # Get settings from config
        self.max_retries = config.llm.max_retries
        self.verbose = config.verbose
        
        # Agent-specific settings
        self.tone = config.agent.writing_tone
        self.include_examples = config.agent.writing_include_examples
        self.max_section_length = config.agent.writing_max_section_length
        self.include_source = config.output.include_source_attribution
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized WritingAgent with {llm_client.provider.value} provider")

    def write(
        self,
        content_plan: ContentPlan,
        parsed_result: Dict,
        include_source_attribution: Optional[bool] = None
    ) -> KBArticle:
        """
        Main writing method - generates KB article
        
        Args:
            content_plan: Content plan from Analysis Agent
            parsed_result: Parsed document with source content
            include_source_attribution: Add "Source:" note at bottom
            
        Returns:
            KBArticle: Complete formatted article
        """
        logger.info("Starting article generation")

        if include_source_attribution is None:
            include_source_attribution = self.include_source
        
        # Validate inputs
        self._validate_inputs(content_plan, parsed_result)
        
        # Extract source content
        source_text = parsed_result['text']
        tables = parsed_result.get('tables', [])
        
        # Build article section by section
        article_content = self._generate_article(
            content_plan,
            source_text,
            tables
        )
        
        # Add source attribution if requested
        if include_source_attribution:
            article_content += self._add_source_attribution(parsed_result)
        
        # Calculate metadata
        word_count = len(article_content.split())
        reading_time = self._calculate_reading_time(word_count)
        
        # Create article object
        article = KBArticle(
            title=content_plan.main_topic,
            content=article_content,
            metadata={
                "document_type": content_plan.document_type,
                "target_audience": content_plan.target_audience,
                "style": content_plan.content_style,
                "sections": len(content_plan.sections),
                "has_tables": len(tables) > 0,
                "source_file": parsed_result.get('metadata', {}).get('filename', 'unknown')
            },
            word_count=word_count,
            estimated_reading_time=reading_time
        )
        
        logger.info(f"Article generation complete: {word_count} words, {len(content_plan.sections)} sections")
        return article
    
    def _validate_inputs(self, content_plan: ContentPlan, parsed_result: Dict):
        """Validate inputs"""
        if not content_plan.sections:
            raise ValueError("Content plan has no sections")
        
        if not parsed_result.get('text'):
            raise ValueError("Parsed result has no text content")

    def _generate_article(
        self,
        content_plan: ContentPlan,
        source_text: str,
        tables: List[Dict]
    ) -> str:
        """
        Generate complete article content
        
        This is the main generation logic that creates the article
        section by section while maintaining strict content fidelity.
        """
        logger.debug("Generating article content")
        
        # Build comprehensive prompt with source content
        user_prompt = self._build_generation_prompt(
            content_plan,
            source_text,
            tables
        )
        
        # Generate article with strict instructions
        try:
            article_content = self.llm.generate(
                prompt=user_prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=8000    # Allow longer articles
            )
            
            return article_content.strip()
            
        except Exception as e:
            logger.error(f"Error generating article: {e}")
            raise


    def _build_generation_prompt(
        self,
        content_plan: ContentPlan,
        source_text: str,
        tables: List[Dict]
    ) -> str:
        """
        Build the generation prompt with all necessary context
        
        This prompt emphasizes using ONLY the source material.
        """
        # Start with critical instruction
        prompt = """CRITICAL: Create a knowledge base article using ONLY the information from the SOURCE DOCUMENT below. 
                    Do NOT add any information from your training data or general knowledge.

                    """
        
        # Add content plan structure
        prompt += f"""ARTICLE STRUCTURE TO FOLLOW:

                    Document Type: {content_plan.document_type}
                    Main Topic: {content_plan.main_topic}
                    Target Audience: {content_plan.target_audience}
                    Content Style: {content_plan.content_style}

                    SECTIONS TO CREATE:
                    """
        
        # Add section structure
        for i, section in enumerate(content_plan.sections, 1):
            prompt += f"\n{i}. {section.title} (Level {section.level})"
            prompt += f"\n   - Summary: {section.summary}"
            prompt += f"\n   - Content elements: {', '.join(section.content_elements)}"
            prompt += f"\n   - Length: {section.estimated_length}"
        
        # Add special instructions
        if content_plan.special_instructions:
            prompt += "\n\nSPECIAL INSTRUCTIONS:\n"
            for instruction in content_plan.special_instructions:
                prompt += f"- {instruction}\n"

        # Add table placement information
        if content_plan.table_placements:
            prompt += "\n\nTABLE PLACEMENTS:\n"
            for placement in content_plan.table_placements:
                prompt += f"- Table {placement.table_index} should go in section '{placement.section_title}'\n"
                prompt += f"  Reason: {placement.placement_reason}\n"
                if placement.should_be_inline:
                    prompt += f"  Format: Include inline as markdown table\n"
                else:
                    prompt += f"  Format: Reference separately\n"
        
        # Add source document - THIS IS THE CRITICAL PART
        prompt += f"\n\n{'='*70}\n"
        prompt += "SOURCE DOCUMENT (USE ONLY THIS INFORMATION):\n"
        prompt += f"{'='*70}\n\n"
        prompt += source_text
        
        # Add tables if present
        if tables:
            prompt += f"\n\n{'='*70}\n"
            prompt += "TABLES FROM SOURCE:\n"
            prompt += f"{'='*70}\n\n"
            
            for i, table in enumerate(tables):
                prompt += f"\nTable {i}:\n"
                if isinstance(table, dict) and 'rows' in table:
                    rows = table['rows']
                    # Format as markdown table preview
                    if rows:
                        prompt += "```\n"
                        for row in rows[:5]:  # Show first 5 rows
                            prompt += " | ".join(str(cell) for cell in row) + "\n"
                        if len(rows) > 5:
                            prompt += f"... ({len(rows) - 5} more rows)\n"
                        prompt += "```\n"
        
        # Final instruction
        prompt += f"\n\n{'='*70}\n"
        prompt += """
                        NOW CREATE THE ARTICLE:

                        1. Follow the section structure provided
                        2. Use ONLY information from the SOURCE DOCUMENT above
                        3. Format as clean markdown with proper headings
                        4. Include tables where specified in the table placements
                        5. Use appropriate markdown formatting (**, *, `, ```, lists, etc.)
                        6. Write in {style} style for {audience} audience
                        7. DO NOT add any information not in the source document

                        Generate the complete article now:
                        """.format(
                                    style=content_plan.content_style,
                                    audience=content_plan.target_audience
                                )
        
        return prompt
    

    def _add_source_attribution(self, parsed_result: Dict) -> str:
        """Add source attribution footer"""
        metadata = parsed_result.get('metadata', {})
        filename = metadata.get('filename', 'source document')
        
        attribution = f"\n\n---\n\n"
        attribution += f"*Source: {filename}*\n"
        
        return attribution
    
    def _calculate_reading_time(self, word_count: int) -> str:
        """
        Calculate estimated reading time
        Average reading speed: 200-250 words per minute
        """
        minutes = max(1, round(word_count / 225))
        
        if minutes == 1:
            return "1 minute"
        elif minutes < 60:
            return f"{minutes} minutes"
        else:
            hours = minutes // 60
            remaining_minutes = minutes % 60
            if remaining_minutes == 0:
                return f"{hours} hour{'s' if hours > 1 else ''}"
            else:
                return f"{hours} hour{'s' if hours > 1 else ''} {remaining_minutes} minutes"
    
    def save_article(
        self,
        article: KBArticle,
        filepath: str,
        include_metadata: bool = True
    ):
        """
        Save article to file
        
        Args:
            article: KBArticle object
            filepath: Path to save markdown file
            include_metadata: Add frontmatter metadata
        """
        content = ""
        
        # Add frontmatter metadata if requested
        if include_metadata:
            content += "---\n"
            content += f"title: {article.title}\n"
            content += f"word_count: {article.word_count}\n"
            content += f"reading_time: {article.estimated_reading_time}\n"
            content += f"document_type: {article.metadata['document_type']}\n"
            content += f"target_audience: {article.metadata['target_audience']}\n"
            content += f"style: {article.metadata['style']}\n"
            content += "---\n\n"
        
        # Add article content
        content += article.content
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Article saved to {filepath}")
    
    def get_article_json(self, article: KBArticle) -> str:
        """Export article as JSON"""
        return json.dumps(article.to_dict(), indent=2, ensure_ascii=False)
    

# ============================================================================
# Utility Functions
# ============================================================================

def create_writing_agent(config: PipelineConfig) -> WritingAgent:
    """
    Convenience function to create WritingAgent with config
    
    Args:
        config: Pipeline configuration
    
    Returns:
        Configured WritingAgent
    """
    llm_client = LLMClient(config=config.llm)
    return WritingAgent(llm_client, config)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\nTo use this agent:")
    print("1. Create config: config = PipelineConfig()")
    print("2. Create agent: agent = create_writing_agent(config)")
    print("3. Get content_plan from Analysis Agent")
    print("4. Get parsed_result from Document Parser")
    print("5. Generate article: article = agent.write(content_plan, parsed_result)")
    print("6. Save article: agent.save_article(article, 'output.md')")
    print("\nExample:")
    print("  from config import PipelineConfig, LLMProvider")
    print("  config = PipelineConfig()")
    print("  config.llm.provider = LLMProvider.GOOGLE")
    print("  agent = create_writing_agent(config)")