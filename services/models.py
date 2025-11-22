"""
Shared data models for KB Generation Pipeline
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


# ============================================================================
# Enums
# ============================================================================
class DocumentType(Enum):
    """Document types for knowledge base articles"""
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    HOW_TO = "how-to"
    CONCEPT = "concept"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"
    GUIDE = "guide"
    API_REFERENCE = "api-reference"
    GENERAL = "general"
    
class DifficultyLevel(Enum):
    """Difficulty levels for articles"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ContentElementType(str, Enum):
    """Types of content elements in the document"""
    PARAGRAPH = "paragraph"
    BULLET_LIST = "bullet_list"
    NUMBERED_LIST = "numbered_list"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    IMAGE = "image"
    QUOTE = "quote"
    CALLOUT = "callout"
    HEADING = "heading"


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


# ============================================================================
# Content Planning Models (used by AnalysisAgent)
# ============================================================================

@dataclass
class Section:
    """Represents a section in the content plan"""
    title: str
    level: int  # 1 = H1, 2 = H2, etc.
    summary: str
    content_elements: List[str]
    estimated_length: str
    subsections: Optional[List['Section']] = None
    
    def to_dict(self) -> Dict:
        result = {
            "title": self.title,
            "level": self.level,
            "summary": self.summary,
            "content_elements": self.content_elements,
            "estimated_length": self.estimated_length
        }
        if self.subsections:
            result["subsections"] = [sub.to_dict() for sub in self.subsections]
        return result


@dataclass
class TablePlacement:
    """Placement information for a table"""
    table_index: int
    section_title: str
    placement_reason: str
    should_be_inline: bool
    formatting_notes: str


@dataclass
class ContentPlan:
    """Complete content plan for the document"""
    document_type: str
    main_topic: str
    target_audience: str
    estimated_reading_time: str
    key_takeaways: List[str]
    sections: List[Section]
    table_placements: List[TablePlacement]
    content_style: str
    special_instructions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "document_type": self.document_type,
            "main_topic": self.main_topic,
            "target_audience": self.target_audience,
            "estimated_reading_time": self.estimated_reading_time,
            "key_takeaways": self.key_takeaways,
            "sections": [section.to_dict() for section in self.sections],
            "table_placements": [asdict(tp) for tp in self.table_placements],
            "content_style": self.content_style,
            "special_instructions": self.special_instructions
        }


# ============================================================================
# Article Models (used by WritingAgent)
# ============================================================================

@dataclass
class KBArticle:
    """Knowledge Base article output"""
    title: str
    content: str
    metadata: Dict[str, Any]
    word_count: int
    estimated_reading_time: str
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "estimated_reading_time": self.estimated_reading_time
        }


# ============================================================================
# Metadata Models (used by MetadataAgent)
# ============================================================================

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