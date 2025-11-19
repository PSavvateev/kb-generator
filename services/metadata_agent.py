# metadata_agent.py
"""
Agent #3: Metadata Generator for KB Generation Pipeline

This agent generates comprehensive metadata for KB articles including:
- SEO keywords and tags
- Categories and topics
- Related articles suggestions
- Difficulty level
- Prerequisites
- Last updated information

Responsibilities:
- Extract key terms and concepts
- Generate relevant tags and keywords
- Classify content by category
- Identify related topics
- Assess content difficulty
- Create search-optimized metadata

Input: 
- KB Article (from writing_agent.py)
- Content plan (from analysis_agent.py)
- Original parsed document (from document_parser.py)

Output: 
- Complete metadata structure (JSON)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from services.llm_client import LLMClient
from services.analysis_agent import ContentPlan
from services.writing_agent import KBArticle

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class DifficultyLevel(str, Enum):
    """Content difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentCategory(str, Enum):
    """Content categories"""
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    REFERENCE = "reference"
    CONCEPT = "concept"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"
    API = "api"
    BEST_PRACTICES = "best_practices"
    GETTING_STARTED = "getting_started"


@dataclass
class RelatedArticle:
    """Related article suggestion"""
    title: str
    relevance_reason: str
    relationship_type: str  # "prerequisite", "follow-up", "related", "alternative"


@dataclass
class ArticleMetadata:
    """Complete metadata for KB article"""
    # Core identification
    title: str
    slug: str  # URL-friendly version of title
    
    # Classification
    category: str
    subcategory: Optional[str]
    tags: List[str]
    keywords: List[str]
    
    # Content characteristics
    difficulty_level: str
    estimated_reading_time: str
    target_audience: str
    
    # SEO and discovery
    meta_description: str
    search_keywords: List[str]
    
    # Relationships
    prerequisites: List[str]
    related_articles: List[RelatedArticle]
    related_topics: List[str]
    
    # Technical metadata
    document_type: str
    content_format: str  # "markdown", "html", etc.
    has_code_examples: bool
    has_tables: bool
    has_images: bool
    
    # Timestamps
    created_date: str
    last_updated: str
    
    # Additional metadata
    author: Optional[str] = None
    version: Optional[str] = None
    language: str = "en"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            "title": self.title,
            "slug": self.slug,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "keywords": self.keywords,
            "difficulty_level": self.difficulty_level,
            "estimated_reading_time": self.estimated_reading_time,
            "target_audience": self.target_audience,
            "meta_description": self.meta_description,
            "search_keywords": self.search_keywords,
            "prerequisites": self.prerequisites,
            "related_articles": [
                {
                    "title": ra.title,
                    "relevance_reason": ra.relevance_reason,
                    "relationship_type": ra.relationship_type
                }
                for ra in self.related_articles
            ],
            "related_topics": self.related_topics,
            "document_type": self.document_type,
            "content_format": self.content_format,
            "has_code_examples": self.has_code_examples,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "created_date": self.created_date,
            "last_updated": self.last_updated,
            "author": self.author,
            "version": self.version,
            "language": self.language
        }
        return result


# ============================================================================
# JSON Schemas for LLM
# ============================================================================

METADATA_EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["tags", "keywords", "category", "difficulty_level", "meta_description"],
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5-10 relevant tags for classification"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "10-15 keywords for search optimization"
        },
        "category": {
            "type": "string",
            "description": "Primary category"
        },
        "subcategory": {
            "type": "string",
            "description": "Subcategory if applicable"
        },
        "difficulty_level": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced", "expert"],
            "description": "Content difficulty level"
        },
        "meta_description": {
            "type": "string",
            "description": "150-160 character SEO description"
        },
        "search_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional search terms"
        }
    }
}

RELATIONSHIPS_SCHEMA = {
    "type": "object",
    "required": ["prerequisites", "related_topics"],
    "properties": {
        "prerequisites": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Required prerequisite knowledge or articles"
        },
        "related_topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Related topics or concepts"
        },
        "related_articles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "relevance_reason", "relationship_type"],
                "properties": {
                    "title": {"type": "string"},
                    "relevance_reason": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": ["prerequisite", "follow-up", "related", "alternative"]
                    }
                }
            }
        }
    }
}

# ============================================================================
# Metadata Agent
# ============================================================================

class MetadataAgent:
    """
    Agent #3: Generates comprehensive metadata for KB articles
    
    This agent analyzes the complete article and generates:
    - Classification metadata (tags, categories, keywords)
    - SEO metadata (descriptions, search terms)
    - Relationship metadata (prerequisites, related content)
    - Technical metadata (format, features, difficulty)
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 3,
        verbose: bool = False
    ):
        """
        Initialize Metadata Agent
        
        Args:
            llm_client: LLM client for generating metadata
            max_retries: Maximum retry attempts
            verbose: Enable verbose logging
        """
        self.llm = llm_client
        self.max_retries = max_retries
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized MetadataAgent with {llm_client.provider} provider")
    
    def generate(
        self,
        article: KBArticle,
        content_plan: ContentPlan,
        parsed_result: Dict,
        author: Optional[str] = None,
        version: Optional[str] = None
    ) -> ArticleMetadata:
        """
        Main generation method - creates complete metadata
        
        Args:
            article: Generated KB article from Writing Agent
            content_plan: Content plan from Analysis Agent
            parsed_result: Original parsed document
            author: Optional author name
            version: Optional version number
            
        Returns:
            ArticleMetadata: Complete metadata structure
        """
        logger.info("Starting metadata generation")
        
        # Step 1: Extract core metadata (tags, keywords, categories)
        core_metadata = self._extract_core_metadata(article, content_plan)
        logger.debug("Core metadata extracted")
        
        # Step 2: Generate relationships (prerequisites, related articles)
        relationships = self._generate_relationships(article, content_plan)
        logger.debug("Relationships generated")
        
        # Step 3: Create SEO metadata
        seo_metadata = self._generate_seo_metadata(article, core_metadata)
        logger.debug("SEO metadata generated")
        
        # Step 4: Analyze technical features
        technical_features = self._analyze_technical_features(article, parsed_result)
        logger.debug("Technical features analyzed")
        
        # Step 5: Generate slug
        slug = self._generate_slug(article.title)
        
        # Step 6: Assemble complete metadata
        metadata = ArticleMetadata(
            title=article.title,
            slug=slug,
            category=core_metadata.get('category', 'general'),
            subcategory=core_metadata.get('subcategory'),
            tags=core_metadata.get('tags', []),
            keywords=core_metadata.get('keywords', []),
            difficulty_level=core_metadata.get('difficulty_level', 'intermediate'),
            estimated_reading_time=article.estimated_reading_time,
            target_audience=content_plan.target_audience,
            meta_description=seo_metadata.get('meta_description', ''),
            search_keywords=seo_metadata.get('search_keywords', []),
            prerequisites=relationships.get('prerequisites', []),
            related_articles=relationships.get('related_articles', []),
            related_topics=relationships.get('related_topics', []),
            document_type=content_plan.document_type,
            content_format='markdown',
            has_code_examples=technical_features['has_code'],
            has_tables=technical_features['has_tables'],
            has_images=technical_features['has_images'],
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            author=author,
            version=version,
            language='en'
        )
        
        logger.info("Metadata generation complete")
        return metadata
    
    def _extract_core_metadata(
        self,
        article: KBArticle,
        content_plan: ContentPlan
    ) -> Dict:
        """
        Step 1: Extract core metadata (tags, keywords, categories)
        """
        logger.debug("Extracting core metadata")
        
        system_prompt = """You are an expert metadata specialist for knowledge base articles.

Your task is to analyze articles and generate comprehensive metadata for:
- Classification (tags, categories)
- Search optimization (keywords)
- User discovery (difficulty level)
- SEO (meta descriptions)

Guidelines:
- Tags should be specific and relevant (5-10 tags)
- Keywords should cover main concepts (10-15 keywords)
- Category should match the content type
- Difficulty level should match the target audience
- Meta description should be compelling and under 160 characters

Be precise and focus on the actual content."""

        user_prompt = f"""Analyze this knowledge base article and generate metadata.

Article Title: {article.title}

Article Content (first 2000 characters):
{article.content[:2000]}

Document Type: {content_plan.document_type}
Target Audience: {content_plan.target_audience}
Content Style: {content_plan.content_style}

Generate:
- tags: 5-10 relevant classification tags
- keywords: 10-15 search keywords covering main concepts
- category: Primary category (tutorial, guide, reference, concept, troubleshooting, faq, api, best_practices, getting_started)
- subcategory: More specific subcategory if applicable
- difficulty_level: beginner, intermediate, advanced, or expert
- meta_description: Compelling 150-160 character description for SEO
- search_keywords: Additional search terms users might use

Provide in JSON format."""

        try:
            metadata = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=METADATA_EXTRACTION_SCHEMA,
                max_retries=self.max_retries
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting core metadata: {e}")
            # Return defaults
            return {
                "tags": ["article", content_plan.document_type],
                "keywords": [article.title.lower()],
                "category": content_plan.document_type,
                "subcategory": None,
                "difficulty_level": "intermediate",
                "meta_description": article.title,
                "search_keywords": []
            }
    
    def _generate_relationships(
        self,
        article: KBArticle,
        content_plan: ContentPlan
    ) -> Dict:
        """
        Step 2: Generate relationship metadata (prerequisites, related articles)
        """
        logger.debug("Generating relationships")
        
        system_prompt = """You are an expert at identifying content relationships and prerequisites.

Your task is to analyze articles and identify:
- Prerequisites: What knowledge/skills readers need first
- Related topics: Connected concepts worth exploring
- Related articles: Suggested follow-up or complementary content

Guidelines:
- Prerequisites should be specific and essential
- Related topics should expand on or complement the content
- Related article suggestions should specify the relationship type:
  * prerequisite: Must read before this article
  * follow-up: Natural next step after this article
  * related: Covers similar or complementary topics
  * alternative: Different approach to same problem

Be practical and user-focused."""

        user_prompt = f"""Identify relationships and prerequisites for this article.

Article Title: {article.title}

Article Content Summary:
Document Type: {content_plan.document_type}
Main Topic: {content_plan.main_topic}
Key Takeaways: {', '.join(content_plan.key_takeaways)}

Target Audience: {content_plan.target_audience}

Sections:
{self._format_sections(content_plan.sections)}

Generate:
- prerequisites: List of required prerequisite knowledge/skills (be specific)
- related_topics: List of related topics users might want to explore
- related_articles: Suggested related articles with:
  * title: Descriptive title of related content
  * relevance_reason: Why it's relevant
  * relationship_type: prerequisite, follow-up, related, or alternative

Provide in JSON format."""

        try:
            relationships = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=RELATIONSHIPS_SCHEMA,
                max_retries=self.max_retries
            )
            
            # Convert related_articles to RelatedArticle objects
            related_articles = []
            for ra in relationships.get('related_articles', []):
                related_articles.append(RelatedArticle(
                    title=ra['title'],
                    relevance_reason=ra['relevance_reason'],
                    relationship_type=ra['relationship_type']
                ))
            
            relationships['related_articles'] = related_articles
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error generating relationships: {e}")
            return {
                "prerequisites": [],
                "related_topics": [],
                "related_articles": []
            }
        
    
    def _generate_seo_metadata(
        self,
        article: KBArticle,
        core_metadata: Dict
    ) -> Dict:
        """
        Step 3: Generate SEO-optimized metadata
        """
        logger.debug("Generating SEO metadata")
        
        # If meta_description is already good, use it
        meta_desc = core_metadata.get('meta_description', '')
        
        # Ensure it's within SEO limits (150-160 chars)
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 50:
            # Generate from title and content
            meta_desc = f"{article.title}. Learn about {core_metadata.get('category', 'this topic')}."
            meta_desc = meta_desc[:160]
        
        # Combine keywords for search
        search_keywords = list(set(
            core_metadata.get('keywords', []) +
            core_metadata.get('search_keywords', []) +
            core_metadata.get('tags', [])
        ))
        
        return {
            "meta_description": meta_desc,
            "search_keywords": search_keywords[:20]  # Limit to 20
        }
    
    def _analyze_technical_features(
        self,
        article: KBArticle,
        parsed_result: Dict
    ) -> Dict:
        """
        Step 4: Analyze technical features of the article
        """
        logger.debug("Analyzing technical features")
        
        content = article.content.lower()
        
        # Check for code blocks
        has_code = (
            '```' in article.content or
            '    ' in article.content  # Indented code
        )
        
        # Check for tables
        has_tables = (
            article.metadata.get('has_tables', False) or
            '|' in article.content  # Markdown tables
        )
        
        # Check for images
        has_images = (
            '![' in article.content or  # Markdown images
            len(parsed_result.get('images', [])) > 0
        )
        
        return {
            "has_code": has_code,
            "has_tables": has_tables,
            "has_images": has_images
        }
    
    def _generate_slug(self, title: str) -> str:
        """
        Generate URL-friendly slug from title
        """
        import re
        
        # Convert to lowercase
        slug = title.lower()
        
        # Replace spaces and special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        # Limit length
        if len(slug) > 60:
            slug = slug[:60].rsplit('-', 1)[0]
        
        return slug
    
    def _format_sections(self, sections) -> str:
        """Format sections for prompt"""
        result = []
        for section in sections:
            result.append(f"- {section.title}: {section.summary}")
        return "\n".join(result)
    

    def get_metadata_json(self, metadata: ArticleMetadata) -> str:
        """
        Export metadata as JSON string
        
        Args:
            metadata: ArticleMetadata object
            
        Returns:
            JSON string representation
        """
        return json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False)
    
    def save_metadata(self, metadata: ArticleMetadata, filepath: str):
        """
        Save metadata to file
        
        Args:
            metadata: ArticleMetadata object
            filepath: Path to save JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.get_metadata_json(metadata))
        
        logger.info(f"Metadata saved to {filepath}")
    
    def generate_frontmatter(self, metadata: ArticleMetadata) -> str:
        """
        Generate YAML frontmatter for markdown files
        
        Args:
            metadata: ArticleMetadata object
            
        Returns:
            YAML frontmatter string
        """
        frontmatter = "---\n"
        frontmatter += f"title: {metadata.title}\n"
        frontmatter += f"slug: {metadata.slug}\n"
        frontmatter += f"category: {metadata.category}\n"
        
        if metadata.subcategory:
            frontmatter += f"subcategory: {metadata.subcategory}\n"
        
        frontmatter += f"difficulty: {metadata.difficulty_level}\n"
        frontmatter += f"reading_time: {metadata.estimated_reading_time}\n"
        frontmatter += f"audience: {metadata.target_audience}\n"
        
        if metadata.tags:
            frontmatter += "tags:\n"
            for tag in metadata.tags:
                frontmatter += f"  - {tag}\n"
        
        if metadata.keywords:
            frontmatter += "keywords:\n"
            for keyword in metadata.keywords[:10]:  # Limit to 10
                frontmatter += f"  - {keyword}\n"
        
        frontmatter += f"description: {metadata.meta_description}\n"
        
        if metadata.prerequisites:
            frontmatter += "prerequisites:\n"
            for prereq in metadata.prerequisites:
                frontmatter += f"  - {prereq}\n"
        
        frontmatter += f"created: {metadata.created_date}\n"
        frontmatter += f"updated: {metadata.last_updated}\n"
        
        if metadata.author:
            frontmatter += f"author: {metadata.author}\n"
        
        if metadata.version:
            frontmatter += f"version: {metadata.version}\n"
        
        frontmatter += "---\n"
        
        return frontmatter
    

# ============================================================================
# Utility Functions
# ============================================================================

def create_metadata_agent(
    provider: str = 'openai',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False
) -> MetadataAgent:
    """
    Convenience function to create MetadataAgent with LLM client
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
        model: Model name (uses default if None)
        api_key: API key (uses env var if None)
        verbose: Enable verbose logging
    
    Returns:
        Configured MetadataAgent
    """
    llm_client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=0.3  # Lower temperature for consistent metadata
    )
    
    return MetadataAgent(llm_client, verbose=verbose)


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
    print("METADATA AGENT - Example Usage")
    print("=" * 70)
    
    # Example: Create agent
    print("\n1. Creating Metadata Agent...")
    try:
        agent = create_metadata_agent(
            provider='openai',
            model='gpt-4',
            verbose=True
        )
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        exit(1)
    
    # Note: In real usage, you would have these from previous agents
    print("\n2. Note: This example requires article, content_plan, and parsed_result")
    print("   from previous pipeline stages (Analysis Agent and Writing Agent)")
    
    print("\n" + "=" * 70)
    print("To use in full pipeline:")
    print("=" * 70)
    print("""
from document_parser import DocumentParser
from analysis_agent import create_analysis_agent
from writing_agent import create_writing_agent
from metadata_agent import create_metadata_agent

# Step 1: Parse document
parser = DocumentParser()
parsed = parser.parse("document.pdf")

# Step 2: Analyze content
analyzer = create_analysis_agent(provider='openai')
content_plan = analyzer.analyze(parsed)

# Step 3: Write article
writer = create_writing_agent(provider='openai')
article = writer.write(content_plan, parsed)

# Step 4: Generate metadata
metadata_gen = create_metadata_agent(provider='openai')
metadata = metadata_gen.generate(
    article=article,
    content_plan=content_plan,
    parsed_result=parsed,
    author="Your Name",
    version="1.0"
)

# Save everything
writer.save_article(article, "article.md")
metadata_gen.save_metadata(metadata, "metadata.json")

# Generate frontmatter
frontmatter = metadata_gen.generate_frontmatter(metadata)
print(frontmatter)
    """)
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)