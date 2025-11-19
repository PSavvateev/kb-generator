"""
Test script for DocumentParser

"""


import json
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from services.document_parser import DocumentParser, parse_document, TableExtractor


# ============================================================================
# Test TableExtractor
# ============================================================================

def test_table_to_markdown():
    """Test table to markdown conversion"""
    extractor = TableExtractor()
    
    table_data = [
        ['Name', 'Age'],
        ['Alice', '25'],
        ['Bob', '30']
    ]
    
    markdown = extractor.table_to_markdown(table_data)
    
    assert '| Name' in markdown
    assert '| Age' in markdown
    assert '| Alice' in markdown
    print("✅ Table to markdown conversion")


def test_table_validation():
    """Test table validation"""
    extractor = TableExtractor()
    
    # Valid table
    valid = [['A', 'B'], ['C', 'D']]
    assert extractor.is_valid_table(valid) == True
    
    # Invalid (too small)
    invalid = [['A']]
    assert extractor.is_valid_table(invalid) == False
    
    # Invalid (empty)
    empty = [['', ''], ['', '']]
    assert extractor.is_valid_table(empty) == False
    
    print("✅ Table validation")


# ============================================================================
# Test Parser Initialization
# ============================================================================

def test_parser_init():
    """Test parser initialization"""
    # Defaults
    parser = DocumentParser()
    assert parser.extract_tables == True
    assert parser.tables_as_markdown == True
    
    # Custom
    parser2 = DocumentParser(extract_tables=False, tables_as_markdown=False)
    assert parser2.extract_tables == False
    assert parser2.tables_as_markdown == False
    
    print("✅ Parser initialization")


# ============================================================================
# Test File Validation
# ============================================================================

def test_file_validation():
    """Test file validation"""
    parser = DocumentParser()
    
    # Non-existent file
    try:
        parser.parse('nonexistent.pdf')
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass
    
    # Unsupported format
    test_file = Path('test.xyz')
    test_file.write_text('test')
    try:
        parser.parse(str(test_file))
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    finally:
        test_file.unlink()
    
    print("✅ File validation")


# ============================================================================
# Test TXT Parsing
# ============================================================================

def test_parse_txt():
    """Test TXT parsing"""
    test_file = Path('test_parse.txt')
    test_content = "Hello world\nThis is a test"
    test_file.write_text(test_content, encoding='utf-8')
    
    try:
        result = parse_document(str(test_file))
        
        assert result['text'] == test_content
        assert result['metadata']['file_type'] == 'txt'
        assert result['metadata']['word_count'] == 6  # "Hello world This is a test" = 6 words
        assert result['metadata']['table_count'] == 0
        assert len(result['tables']) == 0
        
        print("✅ TXT parsing")
    
    finally:
        test_file.unlink()


# ============================================================================
# Test PDF Parsing
# ============================================================================

def test_parse_pdf():
    """Test PDF parsing"""
    test_file = Path('test.pdf')
    
    if not test_file.exists():
        print("⚠️  PDF parsing - SKIPPED (test.pdf not found)")
        return
    
    result = parse_document(str(test_file))
    
    assert result['metadata']['file_type'] == 'pdf'
    assert 'text' in result
    assert 'metadata' in result
    assert 'tables' in result
    assert result['metadata']['page_count'] > 0
    
    print("✅ PDF parsing")


def test_parse_pdf_table_modes():
    """Test PDF table extraction modes"""
    test_file = Path('test.pdf')
    
    if not test_file.exists():
        print("⚠️  PDF table modes - SKIPPED (test.pdf not found)")
        return
    
    # With tables
    result1 = parse_document(str(test_file), extract_tables=True)
    assert 'tables' in result1
    
    # Without tables
    result2 = parse_document(str(test_file), extract_tables=False)
    assert result2['metadata']['table_count'] == 0
    
    print("✅ PDF table extraction modes")


# ============================================================================
# Test DOCX Parsing
# ============================================================================

def test_parse_docx():
    """Test DOCX parsing"""
    test_file = Path('test.docx')
    
    if not test_file.exists():
        print("⚠️  DOCX parsing - SKIPPED (test.docx not found)")
        return
    
    result = parse_document(str(test_file))
    
    assert result['metadata']['file_type'] == 'docx'
    assert 'text' in result
    assert 'metadata' in result
    assert 'tables' in result
    
    print("✅ DOCX parsing")


def test_parse_docx_tables():
    """Test DOCX table extraction"""
    test_file = Path('test.docx')
    
    if not test_file.exists():
        print("⚠️  DOCX tables - SKIPPED (test.docx not found)")
        return
    
    result = parse_document(str(test_file), extract_tables=True)
    
    if result['tables']:
        table = result['tables'][0]
        assert 'data' in table
        assert 'markdown' in table
        assert 'location' in table
        assert table['page'] is None  # DOCX doesn't have page numbers
        
        print(f"✅ DOCX table extraction ({len(result['tables'])} tables found)")
    else:
        print("✅ DOCX table extraction (no tables in document)")


def test_parse_docx_table_modes():
    """Test DOCX tables_as_markdown option"""
    test_file = Path('test.docx')
    
    if not test_file.exists():
        print("⚠️  DOCX table modes - SKIPPED (test.docx not found)")
        return
    
    # Tables in text
    result1 = parse_document(str(test_file), tables_as_markdown=True)
    
    # Tables separate
    result2 = parse_document(str(test_file), tables_as_markdown=False)
    
    if result1['tables']:
        # With tables_as_markdown=True, text should contain markdown tables
        assert '|' in result1['text']
    
    print("✅ DOCX table modes")


# ============================================================================
# Test Output Structure
# ============================================================================

def test_output_structure():
    """Test output structure"""
    test_file = Path('test_structure.txt')
    test_file.write_text('Test content')
    
    try:
        result = parse_document(str(test_file))
        
        # Top-level keys
        assert 'text' in result
        assert 'metadata' in result
        assert 'tables' in result
        
        # Types
        assert isinstance(result['text'], str)
        assert isinstance(result['metadata'], dict)
        assert isinstance(result['tables'], list)
        
        # Metadata fields
        metadata = result['metadata']
        assert 'file_type' in metadata
        assert 'file_size_bytes' in metadata
        assert 'word_count' in metadata
        assert 'char_count' in metadata
        assert 'table_count' in metadata
        
        print("✅ Output structure")
    
    finally:
        test_file.unlink()


def test_table_structure():
    """Test table structure"""
    test_file = Path('test.docx')
    
    if not test_file.exists():
        print("⚠️  Table structure - SKIPPED (test.docx not found)")
        return
    
    result = parse_document(str(test_file))
    
    if result['tables']:
        table = result['tables'][0]
        
        # Required fields
        assert 'data' in table
        assert 'markdown' in table
        assert 'location' in table
        
        # Types
        assert isinstance(table['data'], list)
        assert isinstance(table['markdown'], str)
        
        # 2D array
        if table['data']:
            assert isinstance(table['data'][0], list)
        
        print("✅ Table structure")
    else:
        print("⚠️  Table structure - SKIPPED (no tables in document)")


# ============================================================================
# Test Convenience Function
# ============================================================================

def test_convenience_function():
    """Test parse_document convenience function"""
    test_file = Path('test_conv.txt')
    test_file.write_text('Convenience test')
    
    try:
        # Basic usage
        result1 = parse_document(str(test_file))
        assert result1['text'] == 'Convenience test'
        
        # With options
        result2 = parse_document(
            str(test_file),
            extract_tables=False,
            tables_as_markdown=False
        )
        assert 'text' in result2
        
        print("✅ Convenience function")
    
    finally:
        test_file.unlink()


# ============================================================================
# Test Metadata
# ============================================================================

def test_metadata():
    """Test metadata completeness"""
    test_file = Path('test_meta.txt')
    test_file.write_text('One two three four five')
    
    try:
        result = parse_document(str(test_file))
        metadata = result['metadata']
        
        assert metadata['file_type'] == 'txt'
        assert metadata['word_count'] == 5
        assert metadata['char_count'] == 23
        assert metadata['table_count'] == 0
        assert 'file_size_bytes' in metadata
        
        print("✅ Metadata completeness")
    
    finally:
        test_file.unlink()


# ============================================================================
# Integration Test
# ============================================================================

def test_integration():
    """Test complete workflow"""
    test_file = Path('test_integration.txt')
    test_file.write_text('Integration test\nMultiple lines\nFor testing')
    
    try:
        # Parse
        parser = DocumentParser(extract_tables=True, tables_as_markdown=True)
        result = parser.parse(str(test_file))
        
        # Validate
        assert result['text'] is not None
        assert result['metadata']['word_count'] > 0
        assert isinstance(result['tables'], list)
        
        # Test metadata to dict
        metadata_dict = result['metadata']
        assert isinstance(metadata_dict, dict)
        
        print("✅ Integration test")
    
    finally:
        test_file.unlink()


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("DOCUMENT PARSER TEST SUITE")
    print("="*70)
    print()
    
    tests = [
        test_table_to_markdown,
        test_table_validation,
        test_parser_init,
        test_file_validation,
        test_parse_txt,
        test_parse_pdf,
        test_parse_pdf_table_modes,
        test_parse_docx,
        test_parse_docx_tables,
        test_parse_docx_table_modes,
        test_output_structure,
        test_table_structure,
        test_convenience_function,
        test_metadata,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} - FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} - ERROR: {e}")
            failed += 1
    
    print()
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return passed, failed


if __name__ == "__main__":
    import sys
    
    passed, failed = run_all_tests()
    
    if failed > 0:
        sys.exit(1)