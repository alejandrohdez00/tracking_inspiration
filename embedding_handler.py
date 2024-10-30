"""Handle all embedding-related operations"""
import torch
from sentence_transformers import SentenceTransformer
from utils import get_device, set_seeds
import numpy as np
import sys
import os
import gc
from text_processing import process_raw_text, split_into_sentences
from cache_handling import generate_cache_key, load_from_cache, save_to_cache

def get_context_window(sentences, embeddings, index, window_size=3):
    """Get a window of sentences and their embeddings around an index."""
    start = max(0, index - window_size // 2)
    end = min(len(sentences), index + window_size // 2 + 1)
    
    context_text = ' '.join(sentences[start:end])
    
    # Get the window of embeddings
    window_embeddings = embeddings[start:end]
    
    # Convert embeddings to tensors if they're numpy arrays
    if isinstance(window_embeddings, np.ndarray):
        # If it's a 2D numpy array, convert the whole array
        window_embeddings = torch.from_numpy(window_embeddings)
    elif isinstance(window_embeddings, list):
        # If it's a list of numpy arrays or tensors, convert each element
        window_embeddings = [
            emb if isinstance(emb, torch.Tensor) else torch.from_numpy(emb)
            for emb in window_embeddings
        ]
        # Stack the tensors
        window_embeddings = torch.stack(window_embeddings)
    
    # Check if embeddings are empty
    if isinstance(window_embeddings, list) and len(window_embeddings) == 0:
        raise ValueError("No embeddings found for the specified index range.")
    
    # Calculate mean embedding
    if isinstance(window_embeddings, list):
        context_embedding = torch.mean(torch.stack(window_embeddings), dim=0)
    else:
        context_embedding = torch.mean(window_embeddings, dim=0)
    
    return context_text, context_embedding

def process_and_embed_texts(metadata, raw_dir, model, device, cache_dir):
    """Process and embed texts with proper tensor conversion."""
    sys.stderr.write("Starting process_and_embed_texts...\n")
    embedded_data = []
    
    for _, row in metadata.iterrows():
        sys.stderr.write(f"Processing book: {row['title']}\n")
        file_path = os.path.join(raw_dir, f"{row['id']}_raw.txt")
        
        if os.path.exists(file_path):
            text = process_raw_text(file_path)
            cache_key = generate_cache_key(text, model.get_sentence_embedding_dimension())
            cached_data = load_from_cache(cache_key, cache_dir)
            
            if cached_data is not None:
                sys.stderr.write(f"Loading embeddings from cache for {row['title']}\n")
                # Convert cached embeddings to tensors if needed
                embeddings = cached_data['embeddings']
                if isinstance(embeddings, list) and isinstance(embeddings[0], np.ndarray):
                    embeddings = [torch.from_numpy(emb) for emb in embeddings]
                
                embedded_data.append({
                    'id': row['id'],
                    'title': row['title'],
                    'author': row['author'],
                    'embeddings': embeddings,
                    'original_sentences': cached_data['original_sentences']
                })
                continue
            
            sys.stderr.write(f"Computing new embeddings for {row['title']}\n")
            sentences = split_into_sentences(text)
            
            set_seeds()
            embeddings = model.encode(sentences, device=device)
            
            # Convert numpy arrays to torch tensors
            if isinstance(embeddings, np.ndarray):
                embeddings = [torch.from_numpy(emb) for emb in embeddings]
            
            cache_data = {
                'embeddings': embeddings,
                'original_sentences': sentences
            }
            save_to_cache(cache_key, cache_data, cache_dir)
            
            embedded_data.append({
                'id': row['id'],
                'title': row['title'],
                'author': row['author'],
                'embeddings': embeddings,
                'original_sentences': sentences
            })
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    sys.stderr.write("Finished process_and_embed_texts.\n")
    return embedded_data