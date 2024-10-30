"""Handle style metrics and analysis"""
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
from text_processing import split_into_sentences

def calculate_style_metrics(text):
    """Calculate stylometric features of the text."""
    sentences = split_into_sentences(text)
    words = word_tokenize(text.lower())
    
    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Vocabulary richness (Type-Token Ratio)
    vocab_richness = len(set(words)) / len(words)
    
    # Punctuation frequency
    punctuation = sum(1 for char in text if char in '.,;:!?')
    punct_ratio = punctuation / len(words)
    
    # Function word distribution
    function_words = set(nltk.corpus.stopwords.words('english'))
    func_word_ratio = sum(1 for word in words if word in function_words) / len(words)
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'vocab_richness': vocab_richness,
        'punct_ratio': punct_ratio,
        'func_word_ratio': func_word_ratio
    }

def calculate_ngram_overlap(text1, text2, n=3):
    """Calculate n-gram overlap between two texts."""
    tokens1 = word_tokenize(text1.lower())
    tokens2 = word_tokenize(text2.lower())
    
    ngrams1 = set(tuple(ng) for ng in ngrams(tokens1, n))
    ngrams2 = set(tuple(ng) for ng in ngrams(tokens2, n))
    
    overlap = len(ngrams1.intersection(ngrams2))
    total = len(ngrams1.union(ngrams2))
    
    return overlap / total if total > 0 else 0

def compare_style_metrics(metrics1, metrics2):
    """Compare style metrics between two texts."""
    differences = {}
    for key in metrics1:
        diff = abs(metrics1[key] - metrics2[key])
        differences[key] = diff
    return differences