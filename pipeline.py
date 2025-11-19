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

from services.document_parser import DocumentParser
from services.analysis_agent import create_analysis_agent, ContentPlan
from services.writing_agent import create_writing_agent, KBArticle
from services.metadata_agent import create_metadata_agent, ArticleMetadata

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Configuration and Status
# ============================================================================

class PipelineStage(str, Enum):
    """Pipeline stages"""
    PARSING = "parsing"
    ANALYSIS = "analysis"
    WRITING = "writing"
    METADATA = "metadata"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    # LLM Settings
    provider: str = 'openai'
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    
    # Pipeline Settings
    output_dir: str = "outputs"
    save_intermediates: bool = True  # Save content plan, etc.
    include_source_attribution: bool = True
    include_metadata_frontmatter: bool = True
    
    # Processing Settings
    verbose: bool = False
    max_retries: int = 3
    
    # Article Settings
    author: Optional[str] = None
    version: Optional[str] = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "output_dir": self.output_dir,
            "save_intermediates": self.save_intermediates,
            "include_source_attribution": self.include_source_attribution,
            "include_metadata_frontmatter": self.include_metadata_frontmatter,
            "verbose": self.verbose,
            "max_retries": self.max_retries,
            "author": self.author,
            "version": self.version
        }


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
        logger.info(f"Provider: {self.config.provider}")
        logger.info(f"Output directory: {self.config.output_dir}")
        
        # Initialize components
        self._init_components()
        
        # Create output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline initialized successfully")
    
    def _init_components(self):
        """Initialize all pipeline components"""
        logger.debug("Initializing pipeline components")
        
        # Document Parser
        self.parser = DocumentParser()
        logger.debug("✓ Document Parser initialized")
        
        # Analysis Agent
        self.analyzer = create_analysis_agent(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            verbose=self.config.verbose
        )
        logger.debug("✓ Analysis Agent initialized")
        
        # Writing Agent
        self.writer = create_writing_agent(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            verbose=self.config.verbose
        )
        logger.debug("✓ Writing Agent initialized")
        
        # Metadata Agent
        self.metadata_generator = create_metadata_agent(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            verbose=self.config.verbose
        )
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
            logger.info("\n[Stage 1/4] Parsing document...")
            parsed_result = self._parse_document(document_path)
            logger.info("✓ Document parsed successfully")
            
            # Stage 2: Analyze Content
            logger.info("\n[Stage 2/4] Analyzing content...")
            content_plan = self._analyze_content(parsed_result)
            logger.info("✓ Content analysis complete")
            
            # Stage 3: Write Article
            logger.info("\n[Stage 3/4] Writing KB article...")
            article = self._write_article(content_plan, parsed_result)
            logger.info("✓ Article written successfully")
            
            # Stage 4: Generate Metadata
            logger.info("\n[Stage 4/4] Generating metadata...")
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
            logger.info(f"Metadata saved to: {paths['metadata']}")
            
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
    
    def _analyze_content(self, parsed_result: Dict) -> ContentPlan:
        """Stage 2: Analyze content and create plan"""
        content_plan = self.analyzer.analyze(parsed_result)
        
        # Validate content plan
        if not content_plan.sections:
            raise ValueError("Content plan has no sections")
        
        logger.debug(f"Document type: {content_plan.document_type}")
        logger.debug(f"Main topic: {content_plan.main_topic}")
        logger.debug(f"Sections: {len(content_plan.sections)}")
        
        return content_plan
    
    def _write_article(self, content_plan: ContentPlan, parsed_result: Dict) -> KBArticle:
        """Stage 3: Write KB article"""
        article = self.writer.write(
            content_plan=content_plan,
            parsed_result=parsed_result,
            include_source_attribution=self.config.include_source_attribution
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
        """Stage 4: Generate metadata"""
        metadata = self.metadata_generator.generate(
            article=article,
            content_plan=content_plan,
            parsed_result=parsed_result,
            author=self.config.author,
            version=self.config.version
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
        if self.config.include_metadata_frontmatter:
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
        metadata_path = doc_dir / f"{metadata.slug}_metadata.json"
        self.metadata_generator.save_metadata(metadata, str(metadata_path))
        paths['metadata'] = str(metadata_path)
        logger.debug(f"Saved metadata: {metadata_path}")
        
        # Save content plan if configured
        if self.config.save_intermediates:
            plan_path = doc_dir / f"{metadata.slug}_plan.json"
            self.analyzer.save_content_plan(content_plan, str(plan_path))
            paths['content_plan'] = str(plan_path)
            logger.debug(f"Saved content plan: {plan_path}")
            
            # Save parsed result
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
                "analyzer": f"AnalysisAgent ({self.config.provider})",
                "writer": f"WritingAgent ({self.config.provider})",
                "metadata": f"MetadataAgent ({self.config.provider})"
            },
            "output_directory": str(self.output_path)
        }
    
# ============================================================================
# Convenience Functions
# ============================================================================

def create_pipeline(
    provider: str = 'openai',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: str = "output",
    verbose: bool = False,
    **kwargs
) -> KBPipeline:
    """
    Convenience function to create a configured pipeline
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
        model: Model name (uses default if None)
        api_key: API key (uses env var if None)
        output_dir: Output directory path
        verbose: Enable verbose logging
        **kwargs: Additional PipelineConfig parameters
        
    Returns:
        Configured KBPipeline
    """
    config = PipelineConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        output_dir=output_dir,
        verbose=verbose,
        **kwargs
    )
    
    return KBPipeline(config)


def quick_process(
    document_path: str,
    provider: str = 'openai',
    output_dir: str = "output",
    verbose: bool = False
) -> PipelineResult:
    """
    Quick processing of a single document with minimal configuration
    
    Args:
        document_path: Path to document
        provider: LLM provider
        output_dir: Output directory
        verbose: Enable verbose output
        
    Returns:
        PipelineResult
    """
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
  # Process single document
  python pipeline.py document.pdf
  
  # Process with specific provider
  python pipeline.py document.pdf --provider anthropic
  
  # Process entire directory
  python pipeline.py docs/ --directory
  
  # Process with custom output
  python pipeline.py document.pdf --output my_kb --author "John Doe"
  
  # Verbose mode
  python pipeline.py document.pdf --verbose
        """
    )
    
    parser.add_argument(
        'input',
        help='Input document path or directory'
    )
    
    parser.add_argument(
        '--provider',
        default='google',
        choices=['openai', 'anthropic', 'google', 'ollama'],
        help='LLM provider (default: google)'
    )
    
    parser.add_argument(
        '--model',
        help='Specific model name (uses provider default if not specified)'
    )
    
    parser.add_argument(
        '--api-key',
        help='API key (uses environment variable if not specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '--directory', '-d',
        action='store_true',
        help='Process all documents in directory'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process directory recursively'
    )
    
    parser.add_argument(
        '--author',
        help='Article author name'
    )
    
    parser.add_argument(
        '--version',
        default='1.0',
        help='Article version (default: 1.0)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-intermediates',
        action='store_true',
        help='Do not save intermediate files (content plan, parsed result)'
    )
    
    parser.add_argument(
        '--no-attribution',
        action='store_true',
        help='Do not include source attribution in articles'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        help='File extensions to process (for directory mode)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    config = PipelineConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        output_dir=args.output,
        save_intermediates=not args.no_intermediates,
        include_source_attribution=not args.no_attribution,
        verbose=args.verbose,
        author=args.author,
        version=args.version
    )
    
    # Create pipeline
    pipeline = KBPipeline(config)
    
    # Process input
    if args.directory:
        # Process directory
        results = pipeline.process_directory(
            directory_path=args.input,
            file_extensions=args.extensions,
            recursive=args.recursive,
            continue_on_error=True
        )
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        print(f"\n✓ Processed {successful}/{len(results)} documents successfully")
        
    else:
        # Process single document
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
from pipeline import PipelineConfig, KBPipeline

config = PipelineConfig(
    provider='anthropic',
    model='claude-3-sonnet-20240229',
    output_dir='my_knowledge_base',
    author='John Doe',
    verbose=True,
    save_intermediates=True
)

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
    

    
