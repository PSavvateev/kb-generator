"""
Content Cleaner Service
Cleans and normalizes extracted text

"""

import re
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CleaningOption(Enum):
    """Available cleaning options"""
    REMOVE_ARTIFACTS = "remove_artifacts"
    REMOVE_HEADERS_FOOTERS = "remove_headers_footers"
    NORMALIZE_WHITESPACE = "normalize_whitespace"
    FIX_ENCODING = "fix_encoding"
    REMOVE_DUPLICATES = "remove_duplicate_lines"
    CLEAN_BULLETS = "clean_bullets_and_numbering"


@dataclass
class CleaningStats:
    """Statistics about cleaning operations"""
    original_length: int = 0
    cleaned_length: int = 0
    lines_removed: int = 0
    artifacts_removed: int = 0
    encoding_fixes: int = 0
    duplicates_removed: int = 0
    
    def __str__(self) -> str:
        reduction = self.original_length - self.cleaned_length
        pct = (reduction / self.original_length * 100) if self.original_length > 0 else 0
        return (
            f"CleaningStats:\n"
            f"  Original length: {self.original_length:,} chars\n"
            f"  Cleaned length: {self.cleaned_length:,} chars\n"
            f"  Reduction: {reduction:,} chars ({pct:.1f}%)\n"
            f"  Lines removed: {self.lines_removed}\n"
            f"  Artifacts removed: {self.artifacts_removed}\n"
            f"  Encoding fixes: {self.encoding_fixes}\n"
            f"  Duplicates removed: {self.duplicates_removed}"
        )


@dataclass
class CleaningConfig:
    """Configuration for content cleaning"""
    enabled_options: Set[CleaningOption] = field(default_factory=lambda: {
        CleaningOption.REMOVE_ARTIFACTS,
        CleaningOption.NORMALIZE_WHITESPACE,
        CleaningOption.FIX_ENCODING,
        CleaningOption.REMOVE_DUPLICATES,
        CleaningOption.CLEAN_BULLETS,
    })
    
    # Security limits
    max_text_length: int = 10_000_000  # 10MB of text
    
    # Header/footer removal (disabled by default as it can be aggressive)
    remove_headers_footers: bool = False
    
    # Statistics
    collect_stats: bool = False


class ContentCleaner:
    """Clean and normalize text content"""
    
    # Common header/footer patterns (more conservative)
    HEADER_FOOTER_PATTERNS = [
        r'Page\s+\d+\s+of\s+\d+',  # "Page X of Y"
        r'^\s*\d+\s*/\s*\d+\s*$',  # "1 / 10" alone on line
        r'^\s*-\s*\d+\s*-\s*$',  # "- 5 -" page numbers
    ]
    
    # Patterns for common artifacts
    ARTIFACT_PATTERNS = [
        (r'\x0c', ''),  # Form feed characters
        (r'\u200b', ''),  # Zero-width spaces
        (r'\ufeff', ''),  # BOM
        (r'[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]', ''),  # Control characters (excluding \n, \r, \t)
    ]
    
    # Comprehensive encoding fixes
    ENCODING_FIXES = {
        # UTF-8 mojibake
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'â€"': '–',
        'Â': '',
        'â€¢': '•',
        'â€¦': '...',
        
        # Latin-1 issues
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        'Ã§': 'ç',
        'Ã´': 'ô',
        'Ã®': 'î',
        'Ã»': 'û',
        'Ã«': 'ë',
        'Ã¯': 'ï',
        'Ã¼': 'ü',
        'Ã±': 'ñ',
        
        # More common issues
        'â‚¬': '€',
        'Â©': '©',
        'Â®': '®',
        'â„¢': '™',
    }
    
    # Smart quotes and special characters
    SMART_CHAR_REPLACEMENTS = {
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',
        '–': '-',
        '—': '-',
        '•': '•',
    }
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize ContentCleaner
        
        Args:
            config: Optional cleaning configuration
        """
        self.config = config or CleaningConfig()
        self.stats = CleaningStats() if self.config.collect_stats else None
        
        logger.info(f"ContentCleaner initialized with {len(self.config.enabled_options)} enabled options")
    
    def clean(self, text: str) -> str:
        """
        Clean text by removing artifacts, normalizing spaces, and fixing common issues
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
            
        Raises:
            ValueError: If text is too long or invalid type
        """
        
        # Validate input
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        
        if not text:
            return ""
        
        # Check size limit
        if len(text) > self.config.max_text_length:
            raise ValueError(
                f"Text too long: {len(text):,} chars. "
                f"Maximum allowed: {self.config.max_text_length:,} chars"
            )
        
        # Initialize stats
        if self.stats:
            self.stats.original_length = len(text)
        
        logger.debug(f"Cleaning text: {len(text):,} characters")
        
        # Apply cleaning steps based on configuration
        if CleaningOption.REMOVE_ARTIFACTS in self.config.enabled_options:
            text = self._remove_artifacts(text)
        
        if self.config.remove_headers_footers:
            text = self._remove_headers_footers(text)
        
        if CleaningOption.NORMALIZE_WHITESPACE in self.config.enabled_options:
            text = self._normalize_whitespace(text)
        
        if CleaningOption.FIX_ENCODING in self.config.enabled_options:
            text = self._fix_encoding_issues(text)
        
        if CleaningOption.REMOVE_DUPLICATES in self.config.enabled_options:
            text = self._remove_duplicate_lines(text)
        
        if CleaningOption.CLEAN_BULLETS in self.config.enabled_options:
            text = self._clean_bullets_and_numbering(text)
        
        # Final normalization
        text = text.strip()
        
        # Update stats
        if self.stats:
            self.stats.cleaned_length = len(text)
        
        logger.debug(f"Cleaning complete: {len(text):,} characters")
        
        return text
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove common text artifacts"""
        
        count = 0
        original_text = text
        
        for pattern, replacement in self.ARTIFACT_PATTERNS:
            new_text = re.sub(pattern, replacement, text)
            if new_text != text:
                count += len(text) - len(new_text)
            text = new_text
        
        if self.stats and count > 0:
            self.stats.artifacts_removed += count
            logger.debug(f"Removed {count} artifact characters")
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header and footer patterns"""
        
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check if line matches header/footer patterns
            is_header_footer = False
            for pattern in self.HEADER_FOOTER_PATTERNS:
                if re.search(pattern, line_stripped):
                    is_header_footer = True
                    removed_count += 1
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        if self.stats and removed_count > 0:
            self.stats.lines_removed += removed_count
            logger.debug(f"Removed {removed_count} header/footer lines")
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        
        # Replace multiple spaces with single space (but preserve single newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        
        # Normalize line breaks (max 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        
        fix_count = 0
        
        # Fix comprehensive encoding issues
        for wrong, correct in self.ENCODING_FIXES.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                fix_count += 1
        
        # Fix smart quotes and special characters
        for wrong, correct in self.SMART_CHAR_REPLACEMENTS.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                fix_count += 1
        
        if self.stats and fix_count > 0:
            self.stats.encoding_fixes += fix_count
            logger.debug(f"Applied {fix_count} encoding fixes")
        
        return text
    
    def _remove_duplicate_lines(self, text: str) -> str:
        """Remove consecutive duplicate lines"""
        
        lines = text.split('\n')
        cleaned_lines = []
        prev_line_stripped = None
        duplicate_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Keep line if:
            # 1. It's different from previous (not a duplicate)
            # 2. It's an empty line (preserve paragraph breaks)
            if line_stripped != prev_line_stripped or line_stripped == "":
                cleaned_lines.append(line)
                prev_line_stripped = line_stripped
            else:
                duplicate_count += 1
        
        if self.stats and duplicate_count > 0:
            self.stats.duplicates_removed += duplicate_count
            logger.debug(f"Removed {duplicate_count} duplicate lines")
        
        return '\n'.join(cleaned_lines)
    
    def _clean_bullets_and_numbering(self, text: str) -> str:
        """Clean up bullet points and numbering"""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Unicode bullet characters
        bullet_chars = '•▪▫▸▹►▻⦿⦾○●◆◇■□'
        
        for line in lines:
            # Normalize bullet points - match leading whitespace + bullet + optional whitespace
            line = re.sub(rf'^(\s*)[{re.escape(bullet_chars)}]\s*', r'\1• ', line)
            
            # Normalize numbered lists (ensure consistent spacing)
            line = re.sub(r'^(\s*)(\d+)([.\)])\s*', r'\1\2\3 ', line)
            
            # Normalize lettered lists
            line = re.sub(r'^(\s*)([a-zA-Z])([.\)])\s*', r'\1\2\3 ', line)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_sections(self, text: str, min_heading_length: int = 3, 
                        max_heading_length: int = 100) -> List[Dict[str, str]]:
        """
        Extract sections from text based on common patterns
        
        Args:
            text: Text to analyze
            min_heading_length: Minimum length for a heading
            max_heading_length: Maximum length for a heading
            
        Returns:
            List of sections with 'heading' and 'content' keys
        """
        
        sections = []
        lines = text.split('\n')
        current_section = {"heading": "Introduction", "content": []}
        
        # Patterns that might indicate headings
        heading_patterns = [
            (r'^[A-Z][A-Z\s]{2,}$', 'all_caps'),  # ALL CAPS (at least 3 chars)
            (r'^\d+\.\s+[A-Z]', 'numbered'),  # Numbered headings
            (r'^#{1,6}\s+', 'markdown'),  # Markdown headings
            (r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}$', 'title_case'),  # Title Case
        ]
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                current_section["content"].append(line)
                continue
            
            # Check length constraints
            if not (min_heading_length <= len(line_stripped) <= max_heading_length):
                current_section["content"].append(line)
                continue
            
            # Check if line matches heading patterns
            is_heading = False
            heading_type = None
            
            for pattern, h_type in heading_patterns:
                if re.match(pattern, line_stripped):
                    is_heading = True
                    heading_type = h_type
                    break
            
            if is_heading:
                # Save current section if it has content
                if current_section["content"]:
                    # Filter out empty lines at start/end
                    content_lines = current_section["content"]
                    while content_lines and not content_lines[0].strip():
                        content_lines.pop(0)
                    while content_lines and not content_lines[-1].strip():
                        content_lines.pop()
                    
                    current_section["content"] = '\n'.join(content_lines)
                    sections.append(current_section)
                
                # Start new section
                heading = line_stripped
                # Clean up heading based on type
                if heading_type == 'markdown':
                    heading = re.sub(r'^#{1,6}\s+', '', heading)
                elif heading_type == 'numbered':
                    heading = re.sub(r'^\d+\.\s+', '', heading)
                
                current_section = {
                    "heading": heading,
                    "content": [],
                    "type": heading_type
                }
            else:
                current_section["content"].append(line)
        
        # Add last section
        if current_section["content"]:
            content_lines = current_section["content"]
            while content_lines and not content_lines[0].strip():
                content_lines.pop(0)
            while content_lines and not content_lines[-1].strip():
                content_lines.pop()
            
            current_section["content"] = '\n'.join(content_lines)
            sections.append(current_section)
        
        logger.info(f"Extracted {len(sections)} sections")
        
        return sections
    
    def get_stats(self) -> Optional[CleaningStats]:
        """Get cleaning statistics if enabled"""
        return self.stats
    
    def reset_stats(self):
        """Reset cleaning statistics"""
        if self.config.collect_stats:
            self.stats = CleaningStats()


# Convenience function
def clean_text(text: str, config: Optional[CleaningConfig] = None) -> str:
    """
    Convenience function to clean text
    
    Args:
        text: Text to clean
        config: Optional cleaning configuration
        
    Returns:
        Cleaned text
    """
    cleaner = ContentCleaner(config=config)
    return cleaner.clean(text)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        sample_text = sys.argv[1]
    else:
        sample_text = """
        Page 1 of 10
        
        INTRODUCTION
        
        This is a test document with â€™smart quotesâ€™ and Ã©ncoding issues.
        
        •   Bullet point 1
        ▪   Bullet point 2
        
        This line appears twice.
        This line appears twice.
        
        1.  First item
        2.  Second item
        
        â€"â€" Extra dashes â€"â€"
        
        \x0c
        
        Page 2 of 10
        """
    
    # Create cleaner with stats
    config = CleaningConfig(collect_stats=True)
    cleaner = ContentCleaner(config=config)
    
    print("="*60)
    print("ORIGINAL TEXT:")
    print("="*60)
    print(sample_text)
    
    print("\n" + "="*60)
    print("CLEANED TEXT:")
    print("="*60)
    cleaned = cleaner.clean(sample_text)
    print(cleaned)
    
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    print(cleaner.get_stats())
    
    print("\n" + "="*60)
    print("SECTIONS:")
    print("="*60)
    sections = cleaner.extract_sections(cleaned)
    for i, section in enumerate(sections, 1):
        print(f"\n{i}. {section['heading']}")
        print(f"   Type: {section.get('type', 'N/A')}")
        print(f"   Content preview: {section['content'][:100]}...")
