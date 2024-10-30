import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from utils import get_device

def load_metadata(file_path):
    """Load metadata from CSV file."""
    return pd.read_csv(file_path)

def process_raw_text(file_path):
    """Process raw text file, removing non-book content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    start = content.find("*** START OF")
    end = content.find("*** END OF")
    if start != -1 and end != -1:
        return content[start:end].strip()
    return content

def embed_sentences(text, model):
    """Embed sentences using the provided model."""
    sentences = text.split('.')
    return model.encode(sentences, device=get_device())

def process_and_embed_texts(metadata, raw_dir, model):
    """Process and embed all texts."""
    embedded_data = []
    for _, row in metadata.iterrows():
        file_path = os.path.join(raw_dir, f"PG{row['id']}_raw.txt")
        if os.path.exists(file_path):
            text = process_raw_text(file_path)
            embeddings = embed_sentences(text, model)
            embedded_data.append({
                'id': row['id'],
                'title': row['title'],
                'author': row['author'],
                'embeddings': embeddings
            })
    return embedded_data
