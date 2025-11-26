"""
KB Generation Pipeline Engine

This is the main orchestrator that runs the complete KB generation pipeline:
1. Document Parser - Extracts content from documents
2. Analysis Agent - Creates content plan
3. Writing Agent - Generates KB article
4. Metadata Agent - Generates metadata

The pipeline provides:
- Sequential execution of all agents
- Error handling and recovery
- Progress tracking
- Batch processing
- Result validation
- Output management
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config import PipelineConfig, LLMProvider, load_env_file
from services.models import ContentPlan, KBArticle, ArticleMetadata
from services.document_parser import DocumentParser
from services.content_cleaner import create_content_cleaner
from services.analysis_agent import create_analysis_agent
from services.writing_agent import create_writing_agent
from services.metadata_agent import create_metadata_agent

# Load environment
load_env_file()

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Configuration and Status
# ============================================================================

class PipelineStage(str, Enum):
    """Pipeline stages"""
    PARSING = "parsing"
    CLEANING = "cleaning" 
    ANALYSIS = "analysis"
    WRITING = "writing"
    METADATA = "metadata"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Result from pipeline execution"""
    success: bool
    stage_completed: str
    
    # Outputs
    parsed_document: Optional[Dict] = None
    content_plan: Optional[ContentPlan] = None
    article: Optional[KBArticle] = None
    metadata: Optional[ArticleMetadata] = None
    
    # Paths to saved files
    article_path: Optional[str] = None
    metadata_path: Optional[str] = None
    content_plan_path: Optional[str] = None
    
    # Execution info
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            "success": self.success,
            "stage_completed": self.stage_completed,
            "execution_time": self.execution_time,
            "article_path": self.article_path,
            "metadata_path": self.metadata_path,
            "content_plan_path": self.content_plan_path
        }
        
        if self.error_message:
            result["error_message"] = self.error_message
        
        if self.metadata:
            result["metadata_summary"] = {
                "title": self.metadata.title,
                "category": self.metadata.category,
                "difficulty": self.metadata.difficulty_level,
                "reading_time": self.metadata.estimated_reading_time,
                "tags": self.metadata.tags[:5]  # First 5 tags
            }
        
        return result
    

# ============================================================================
# Pipeline Engine
# ============================================================================

class KBPipeline:
    """
    Main KB Generation Pipeline
    
    Orchestrates the complete workflow:
    Parse → Analyze → Write → Generate Metadata
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Set up logging
        if self.config.verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        logger.info("Initializing KB Generation Pipeline")
        logger.info(f"Provider: {self.config.llm.provider.value}")
        logger.info(f"Model: {self.config.llm.get_model()}")
        logger.info(f"Output directory: {self.config.output.output_dir}")
        
        # Initialize components
        self._init_components()
        
        # Create output directory
        self.output_path = Path(self.config.output.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline initialized successfully")


    def _init_components(self):
        """Initialize all pipeline components"""
        logger.debug("Initializing pipeline components")
        
        # Document Parser (Stage 1)
        self.parser = DocumentParser()
        logger.debug("✓ Document Parser initialized")

        # Content Cleaner (Stage 2) 
        if self.config.cleaner.enabled:
            self.cleaner = create_content_cleaner(self.config)
            logger.debug("✓ Content Cleaner initialized")
        else:
            self.cleaner = None
            logger.debug("⊘ Content Cleaner disabled")
        
        # Analysis Agent (Stage 3)
        self.analyzer = create_analysis_agent(self.config)
        logger.debug("✓ Analysis Agent initialized")
        
        # Writing Agent (Stage 4)
        self.writer = create_writing_agent(self.config)
        logger.debug("✓ Writing Agent initialized")
        
        # Metadata Agent (Stage 5)
        self.metadata_generator = create_metadata_agent(self.config)
        logger.debug("✓ Metadata Agent initialized")


    def process_document(self, document_path: str) -> PipelineResult:
        """
        Process a single document through the complete pipeline
        
        Args:
            document_path: Path to document file
            
        Returns:
            PipelineResult with outputs and status
        """
        start_time = datetime.now()
        
        logger.info("=" * 70)
        logger.info(f"Starting pipeline for: {document_path}")
        logger.info("=" * 70)
        
        try:
            # Stage 1: Parse Document
            logger.info("\n[Stage 1/5] Parsing document...")
            parsed_result = self._parse_document(document_path)
            logger.info("✓ Document parsed successfully")

            # Stage 2: Clean Content ⭐ ADD THIS ENTIRE BLOCK
            if self.config.cleaner.enabled:
                logger.info("\n[Stage 2/5] Cleaning content...")
                parsed_result = self._clean_content(parsed_result)
                logger.info("✓ Content cleaned successfully")
            else:
                logger.info("\n[Stage 2/5] Cleaning disabled - skipping")
            
            # Stage 3: Analyze Content
            logger.info("\n[Stage 3/5] Analyzing content...")
            content_plan = self._analyze_content(parsed_result)
            logger.info("✓ Content analysis complete")
            
            # Stage 4: Write Article
            logger.info("\n[Stage 4/5] Writing KB article...")
            article = self._write_article(content_plan, parsed_result)
            logger.info("✓ Article written successfully")
            
            # Stage 4: Generate Metadata
            logger.info("\n[Stage 5/5] Generating metadata...")
            metadata = self._generate_metadata(article, content_plan, parsed_result)
            logger.info("✓ Metadata generated successfully")
            
            # Save outputs
            logger.info("\nSaving outputs...")
            paths = self._save_outputs(
                article=article,
                metadata=metadata,
                content_plan=content_plan,
                parsed_result=parsed_result,
                source_filename=Path(document_path).stem
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info(f"Article saved to: {paths['article']}")
            if 'metadata' in paths:
                logger.info(f"Metadata saved to: {paths['metadata']}")
            if 'content_plan' in paths:
                logger.info(f"Content plan saved to: {paths['content_plan']}")
            
            return PipelineResult(
                success=True,
                stage_completed=PipelineStage.COMPLETE.value,
                parsed_document=parsed_result,
                content_plan=content_plan,
                article=article,
                metadata=metadata,
                article_path=paths['article'],
                metadata_path=paths['metadata'],
                content_plan_path=paths.get('content_plan'),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"\n❌ Pipeline failed: {str(e)}")
            
            import traceback
            if self.config.verbose:
                traceback.print_exc()
            
            return PipelineResult(
                success=False,
                stage_completed=PipelineStage.FAILED.value,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _parse_document(self, document_path: str) -> Dict:
        """Stage 1: Parse document"""
        parsed = self.parser.parse(document_path)
        
        # Validate parsed result
        if not parsed.get('text'):
            raise ValueError("Parsed document contains no text")
        
        logger.debug(f"Extracted {len(parsed['text'])} characters")
        logger.debug(f"Found {len(parsed.get('tables', []))} tables")
        logger.debug(f"Found {len(parsed.get('images', []))} images")
        
        return parsed
    
    def _clean_content(self, parsed_result: Dict) -> Dict:  # ⭐ ADD THIS METHOD
        """
        Stage 2: Clean extracted content
        
        Removes artifacts, fixes encoding, normalizes whitespace, etc.
        """
        logger.debug("Cleaning extracted content")
        
        if not self.cleaner:
            logger.warning("Cleaner not initialized but cleaning requested")
            return parsed_result
        
        original_text = parsed_result['text']
        original_length = len(original_text)
        
        # Clean the text
        cleaned_text = self.cleaner.clean(original_text)
        cleaned_length = len(cleaned_text)
        
        # Calculate reduction
        reduction = original_length - cleaned_length
        reduction_pct = (reduction / original_length * 100) if original_length > 0 else 0
        
        logger.debug(f"Text cleaned: {original_length:,} → {cleaned_length:,} chars")
        logger.debug(f"Reduction: {reduction:,} chars ({reduction_pct:.1f}%)")
        
        # Log statistics if enabled
        if self.config.cleaner.collect_stats:
            stats = self.cleaner.get_stats()
            if stats:
                logger.info("Cleaning statistics:")
                logger.info(f"  Artifacts removed: {stats.artifacts_removed}")
                logger.info(f"  Encoding fixes: {stats.encoding_fixes}")
                logger.info(f"  Duplicates removed: {stats.duplicates_removed}")
                logger.info(f"  Lines removed: {stats.lines_removed}")
        
        # Update parsed result with cleaned text
        parsed_result['text'] = cleaned_text
        
        # Validate cleaned result
        if not cleaned_text.strip():
            raise ValueError("Cleaned text is empty - cleaning may have been too aggressive")
        
        return parsed_result
    
    def _analyze_content(self, parsed_result: Dict) -> ContentPlan:
        """Stage 3: Analyze content and create plan"""
        content_plan = self.analyzer.analyze(parsed_result)
        
        # Validate content plan
        if not content_plan.sections:
            raise ValueError("Content plan has no sections")
        
        logger.debug(f"Document type: {content_plan.document_type}")
        logger.debug(f"Main topic: {content_plan.main_topic}")
        logger.debug(f"Sections: {len(content_plan.sections)}")
        
        return content_plan
    
    def _write_article(self, content_plan: ContentPlan, parsed_result: Dict) -> KBArticle:
        """Stage 4: Write KB article"""
        article = self.writer.write(
            content_plan=content_plan,
            parsed_result=parsed_result
            # include_source_attribution now comes from config
        )
        
        # Validate article
        if not article.content:
            raise ValueError("Generated article is empty")
        
        logger.debug(f"Article title: {article.title}")
        logger.debug(f"Word count: {article.word_count}")
        logger.debug(f"Reading time: {article.estimated_reading_time}")
        
        return article

    def _generate_metadata(
        self,
        article: KBArticle,
        content_plan: ContentPlan,
        parsed_result: Dict
    ) -> ArticleMetadata:
        """Stage 5: Generate metadata"""
        metadata = self.metadata_generator.generate(
            article=article,
            content_plan=content_plan,
            parsed_result=parsed_result
        
        )
        
        logger.debug(f"Category: {metadata.category}")
        logger.debug(f"Tags: {metadata.tags}")
        logger.debug(f"Difficulty: {metadata.difficulty_level}")
        
        return metadata
    
    def _save_outputs(
        self,
        article: KBArticle,
        metadata: ArticleMetadata,
        content_plan: ContentPlan,
        parsed_result: Dict,
        source_filename: str
    ) -> Dict[str, str]:
        """Save all outputs to files"""
        paths = {}
        
        # Create subdirectory for this document
        doc_dir = self.output_path / source_filename
        doc_dir.mkdir(exist_ok=True)
        
        # Save article
        article_path = doc_dir / f"{metadata.slug}.md"
        
        # Add frontmatter if configured
        if self.config.output.include_frontmatter:
            frontmatter = self.metadata_generator.generate_frontmatter(metadata)
            full_content = frontmatter + "\n" + article.content
            
            with open(article_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
        else:
            self.writer.save_article(
                article,
                str(article_path),
                include_metadata=True
            )
        
        paths['article'] = str(article_path)
        logger.debug(f"Saved article: {article_path}")
        
        # Save metadata
        if self.config.output.generate_metadata:
            metadata_path = doc_dir / f"{metadata.slug}_metadata.json"
            self.metadata_generator.save_metadata(metadata, str(metadata_path))
            paths['metadata'] = str(metadata_path)
            logger.debug(f"Saved metadata: {metadata_path}")
        
        # Save content plan if configured
        if self.config.output.generate_plan:
            plan_path = doc_dir / f"{metadata.slug}_plan.json"
            self.analyzer.save_content_plan(content_plan, str(plan_path))
            paths['content_plan'] = str(plan_path)
            logger.debug(f"Saved content plan: {plan_path}")
        
        # Save parsed result if configured
        if self.config.output.generate_parsed:
            parsed_path = doc_dir / f"{metadata.slug}_parsed.json"
            with open(parsed_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_result, f, indent=2, ensure_ascii=False)
            paths['parsed'] = str(parsed_path)
            logger.debug(f"Saved parsed result: {parsed_path}")
        
        return paths


    def process_batch(
        self,
        document_paths: List[str],
        continue_on_error: bool = True
    ) -> List[PipelineResult]:
        """
        Process multiple documents through the pipeline
        
        Args:
            document_paths: List of document file paths
            continue_on_error: Continue processing if one document fails
            
        Returns:
            List of PipelineResult objects
        """
        logger.info("=" * 70)
        logger.info(f"Starting batch processing: {len(document_paths)} documents")
        logger.info("=" * 70)
        
        results = []
        successful = 0
        failed = 0
        
        for i, doc_path in enumerate(document_paths, 1):
            logger.info(f"\n[Document {i}/{len(document_paths)}]")
            
            try:
                result = self.process_document(doc_path)
                results.append(result)
                
                if result.success:
                    successful += 1
                else:
                    failed += 1
                    if not continue_on_error:
                        logger.error("Stopping batch processing due to error")
                        break
                        
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                failed += 1
                
                results.append(PipelineResult(
                    success=False,
                    stage_completed=PipelineStage.FAILED.value,
                    error_message=str(e)
                ))
                
                if not continue_on_error:
                    logger.error("Stopping batch processing due to error")
                    break
        
        logger.info("\n" + "=" * 70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total documents: {len(document_paths)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        return results
    
    def process_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = False,
        continue_on_error: bool = True
    ) -> List[PipelineResult]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory
            file_extensions: List of file extensions to process (None = all supported)
            recursive: Process subdirectories
            continue_on_error: Continue if one document fails
            
        Returns:
            List of PipelineResult objects
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Default supported extensions
        if file_extensions is None:
            file_extensions = ['.pdf', '.txt', '.docx', '.md', '.html']
        
        # Find all matching files
        if recursive:
            document_paths = []
            for ext in file_extensions:
                document_paths.extend(directory.rglob(f"*{ext}"))
        else:
            document_paths = []
            for ext in file_extensions:
                document_paths.extend(directory.glob(f"*{ext}"))
        
        document_paths = [str(p) for p in document_paths]
        
        logger.info(f"Found {len(document_paths)} documents in {directory_path}")
        
        if not document_paths:
            logger.warning("No documents found to process")
            return []
        
        return self.process_batch(document_paths, continue_on_error)
    
    def get_pipeline_info(self) -> Dict:
        """Get information about pipeline configuration"""
        return {
            "config": self.config.to_dict(),
            "components": {
                "parser": "DocumentParser",
                "analyzer": f"AnalysisAgent ({self.config.llm.provider.value})",
                "writer": f"WritingAgent ({self.config.llm.provider.value})",
                "metadata": f"MetadataAgent ({self.config.llm.provider.value})"
            },
            "output_directory": str(self.output_path)
        }
    

# ============================================================================
# Convenience Functions
# ============================================================================

def create_pipeline(
    provider: str = 'google',
    model: Optional[str] = None,
    output_dir: str = "outputs",
    verbose: bool = False,
    **kwargs
) -> KBPipeline:
    """
    Convenience function to create a configured pipeline
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
        model: Model name (uses default if None)
        output_dir: Output directory path
        verbose: Enable verbose logging
        **kwargs: Additional config parameters
        
    Returns:
        Configured KBPipeline
    """
    config = PipelineConfig()
    config.llm.provider = LLMProvider(provider)
    if model:
        config.llm.model = model
    config.output.output_dir = output_dir
    config.verbose = verbose
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return KBPipeline(config)


def quick_process(
    document_path: str,
    provider: str = 'google',
    output_dir: str = "outputs",
    verbose: bool = False
) -> PipelineResult:
    """Quick processing of a single document"""
    pipeline = create_pipeline(
        provider=provider,
        output_dir=output_dir,
        verbose=verbose
    )
    
    return pipeline.process_document(document_path)


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point for CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KB Generation Pipeline - Convert documents to knowledge base articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py document.pdf
  python pipeline.py document.pdf --provider anthropic
  python pipeline.py docs/ --directory
  python pipeline.py document.pdf --output my_kb --author "John Doe"
  python pipeline.py document.pdf --verbose
  python pipeline.py document.pdf --no-cleaning  
  python pipeline.py document.pdf --remove-headers  
  python pipeline.py document.pdf --cleaning-stats  
        """
    )
    
    parser.add_argument('input', help='Input document path or directory')
    parser.add_argument('--provider', default='google', 
                       choices=['openai', 'anthropic', 'google', 'ollama'])
    parser.add_argument('--model', help='Specific model name')
    parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    parser.add_argument('--directory', '-d', action='store_true')
    parser.add_argument('--recursive', '-r', action='store_true')
    parser.add_argument('--author', help='Article author name')
    parser.add_argument('--version', default='1.0', help='Article version')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-plan', action='store_true', 
                       help='Do not save content plan')
    parser.add_argument('--no-attribution', action='store_true',
                       help='Do not include source attribution')
    parser.add_argument('--extensions', nargs='+',
                       help='File extensions to process')
    
    parser.add_argument('--no-cleaning', action='store_true',
                       help='Disable content cleaning stage')
    parser.add_argument('--remove-headers', action='store_true',
                       help='Enable header/footer removal (aggressive)')
    parser.add_argument('--cleaning-stats', action='store_true',
                       help='Collect and display cleaning statistics')
    

    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig()
    config.llm.provider = LLMProvider(args.provider)
    if args.model:
        config.llm.model = args.model
    config.output.output_dir = args.output
    config.output.generate_plan = not args.no_plan
    config.output.include_source_attribution = not args.no_attribution
    config.verbose = args.verbose
    if args.author:
        config.output.author = args.author
    if args.version:
        config.output.version = args.version

    config.cleaner.enabled = not args.no_cleaning
    config.cleaner.remove_headers_footers = args.remove_headers
    config.cleaner.collect_stats = args.cleaning_stats
    
    # Create pipeline
    pipeline = KBPipeline(config)
    
    # Process input
    if args.directory:
        results = pipeline.process_directory(
            directory_path=args.input,
            file_extensions=args.extensions,
            recursive=args.recursive,
            continue_on_error=True
        )
        successful = sum(1 for r in results if r.success)
        print(f"\n✓ Processed {successful}/{len(results)} documents successfully")
    else:
        result = pipeline.process_document(args.input)
        if result.success:
            print(f"\n✓ Article created: {result.article_path}")
            print(f"✓ Metadata saved: {result.metadata_path}")
        else:
            print(f"\n✗ Processing failed: {result.error_message}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


# ============================================================================
# Example Usage
# ============================================================================

"""
EXAMPLE USAGE PATTERNS:

1. Basic Single Document Processing:
-----------------------------------
from pipeline import create_pipeline

pipeline = create_pipeline(provider='openai')
result = pipeline.process_document('document.pdf')

if result.success:
    print(f"Article: {result.article_path}")
    print(f"Metadata: {result.metadata_path}")


2. Process with Custom Configuration:
------------------------------------
from config import PipelineConfig, LLMProvider
from pipeline import KBPipeline

config = PipelineConfig()
config.llm.provider = LLMProvider.ANTHROPIC
config.llm.model = 'claude-3-5-sonnet-20241022'
config.output.output_dir = 'my_knowledge_base'
config.output.author = 'John Doe'
config.verbose = True
config.output.generate_plan = True
config.output.generate_parsed = True

pipeline = KBPipeline(config)
result = pipeline.process_document('tutorial.pdf')


3. Batch Processing:
------------------
pipeline = create_pipeline(provider='openai', output_dir='kb_articles')

documents = ['doc1.pdf', 'doc2.docx', 'doc3.txt']
results = pipeline.process_batch(documents)

for result in results:
    if result.success:
        print(f"✓ {result.metadata.title}")
    else:
        print(f"✗ Failed: {result.error_message}")


4. Process Entire Directory:
---------------------------
pipeline = create_pipeline(provider='google')

results = pipeline.process_directory(
    directory_path='documents/',
    file_extensions=['.pdf', '.docx'],
    recursive=True
)

print(f"Processed {len(results)} documents")


5. Quick Processing (Minimal Setup):
-----------------------------------
from pipeline import quick_process

result = quick_process('document.pdf', provider='openai')

if result.success:
    print(f"Done! Check: {result.article_path}")


6. Access Generated Content:
---------------------------
result = pipeline.process_document('doc.pdf')

if result.success:
    # Access the article
    article = result.article
    print(f"Title: {article.title}")
    print(f"Word count: {article.word_count}")
    
    # Access metadata
    metadata = result.metadata
    print(f"Category: {metadata.category}")
    print(f"Tags: {metadata.tags}")
    print(f"Difficulty: {metadata.difficulty_level}")
    
    # Access content plan
    plan = result.content_plan
    print(f"Document type: {plan.document_type}")
    print(f"Sections: {len(plan.sections)}")


7. Command Line Usage:
---------------------
# Process single document
python pipeline.py document.pdf --provider openai --verbose

# Process directory
python pipeline.py documents/ --directory --recursive --output kb_output

# With custom settings
python pipeline.py doc.pdf --author "Jane Smith" --version "2.0" --output my_kb
"""
    

    
