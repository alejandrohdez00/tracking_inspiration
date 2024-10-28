import os
import pickle
import hashlib
from datetime import datetime
import sys

def get_cache_path(output_dir):
    """Create and return the cache directory path."""
    cache_dir = os.path.join(output_dir, 'embedding_cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def generate_cache_key(text_content, model_name):
    """Generate a unique cache key based on text content and model."""
    content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
    return f"{content_hash}_{model_name}"

def save_to_cache(cache_key, data, cache_dir):
    """Save embedding data to cache."""
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Save metadata for cache management
    metadata_file = os.path.join(cache_dir, f"{cache_key}_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Created: {datetime.now()}\n")
        f.write(f"Embedding shape: {data['embeddings'][0].shape}\n")
        f.write(f"Number of sentences: {len(data['original_sentences'])}\n")

def load_from_cache(cache_key, cache_dir):
    """Load embedding data from cache if it exists."""
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            sys.stderr.write(f"Error loading cache: {str(e)}\n")
            return None
    return None

def clean_cache(cache_dir, max_age_days=30):
    """Remove cache files older than specified days."""
    sys.stderr.write("Cleaning old cache files...\n")
    now = datetime.now()
    
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        if os.path.isfile(file_path):
            file_age = datetime.fromtimestamp(os.path.getctime(file_path))
            age_days = (now - file_age).days
            
            if age_days > max_age_days:
                try:
                    os.remove(file_path)
                    sys.stderr.write(f"Removed old cache file: {filename}\n")
                except Exception as e:
                    sys.stderr.write(f"Error removing cache file {filename}: {str(e)}\n")