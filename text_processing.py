"""Handle all text processing related functions"""
import re
import nltk
from nltk.tokenize import word_tokenize

def process_raw_text(file_path):
    """Process raw text file, removing non-book content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    start = content.find("*** START OF")
    end = content.find("*** END OF")
    if start != -1 and end != -1:
        content = content[start:end].strip()
    
    content = '\n'.join(line for line in content.split('\n') if not line.strip().startswith('_'))
    return content

def split_into_sentences(text):
    """
    Split text into sentences while preserving honorifics, abbreviations, and handling edge cases.
    Returns a list of cleaned sentences.
    """
    # Common abbreviations and titles that include periods
    abbreviations = {
        # Titles
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'Rev.',
        # Business
        'Ltd.', 'Co.', 'Corp.', 'Inc.', 'LLC.',
        # Locations
        'St.', 'Ave.', 'Blvd.', 'Rd.', 'Apt.',
        # Common abbreviations
        'etc.', 'i.e.', 'e.g.', 'vs.', 'viz.', 'al.',
        # Academic
        'Ph.D.', 'B.A.', 'M.A.', 'D.D.S.',
        # Geography
        'U.S.A.', 'U.K.', 'E.U.',
        # Time
        'a.m.', 'p.m.',
        # Add more as needed
    }
    
    # Sort abbreviations by length (longest first) to avoid partial matches
    sorted_abbreviations = sorted(abbreviations, key=len, reverse=True)
    
    # Replace periods in abbreviations with a unique marker
    # Using a marker unlikely to appear in normal text
    marker = '<!PERIOD!>'
    text_copy = text
    
    # Replace periods in abbreviations
    for abbr in sorted_abbreviations:
        # Use word boundary to avoid matching partial words
        pattern = r'\b' + re.escape(abbr)
        text_copy = re.sub(pattern, abbr.replace('.', marker), text_copy)
    
    # Handle decimal numbers
    text_copy = re.sub(r'(\d+)\.(\d+)', r'\1' + marker + r'\2', text_copy)
    
    # Handle ellipsis
    text_copy = text_copy.replace('...', '<!ELLIPSIS!>')
    
    # Split on sentence boundaries
    # Looking for:
    # 1. Period, exclamation mark, or question mark
    # 2. Followed by space or newline
    # 3. Followed by capital letter or number
    sentence_boundaries = r'(?<=[.!?])(?=\s+(?:[A-Z]|\d))|(?<=[.!?])(?=\n)|(?<=[.!?])(?=\s*$)'
    
    sentences = re.split(sentence_boundaries, text_copy)
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        # Restore periods in abbreviations
        sentence = sentence.replace(marker, '.')
        # Restore ellipsis
        sentence = sentence.replace('<!ELLIPSIS!>', '...')
        # Remove extra whitespace
        sentence = ' '.join(sentence.split())
        
        if sentence.strip():
            cleaned_sentences.append(sentence.strip())
    
    # Additional validation to merge incomplete sentences
    merged_sentences = []
    current_sentence = ''
    
    for sentence in cleaned_sentences:
        current_sentence = current_sentence + ' ' + sentence if current_sentence else sentence
        
        # Check if this forms a complete sentence
        # A complete sentence should either:
        # 1. End with terminal punctuation
        # 2. Or be followed by a sentence starting with a capital letter
        if re.search(r'[.!?]$', current_sentence.strip()):
            merged_sentences.append(current_sentence.strip())
            current_sentence = ''
    
    # Add any remaining content
    if current_sentence.strip():
        merged_sentences.append(current_sentence.strip())
    
    return merged_sentences