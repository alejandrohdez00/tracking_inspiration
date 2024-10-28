import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import torch
from utils import get_device, ensure_dir
from cache_handling import get_cache_path, generate_cache_key, save_to_cache, load_from_cache
import argparse
import sys
import gc
from openai import OpenAI
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
import re
import random


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def calculate_ngram_overlap(text1, text2, n=3):
    """Calculate n-gram overlap between two texts."""
    tokens1 = word_tokenize(text1.lower())
    tokens2 = word_tokenize(text2.lower())
    
    ngrams1 = set(tuple(ng) for ng in ngrams(tokens1, n))
    ngrams2 = set(tuple(ng) for ng in ngrams(tokens2, n))
    
    overlap = len(ngrams1.intersection(ngrams2))
    total = len(ngrams1.union(ngrams2))
    
    return overlap / total if total > 0 else 0

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

def compare_style_metrics(metrics1, metrics2):
    """Compare style metrics between two texts."""
    differences = {}
    for key in metrics1:
        diff = abs(metrics1[key] - metrics2[key])
        differences[key] = diff
    return differences

def analyze_similarity(data, generated_plot, plot_embeddings):
    """Analyze similarity between generated and original texts using multiple metrics."""
    sys.stderr.write("Starting comprehensive similarity analysis...\n")
    analyzed_data = []
    generated_sentences = split_into_sentences(generated_plot)
    
    for item in data:
        sys.stderr.write(f"Analyzing similarities for '{item['title']}'...\n")
        
        # Embedding-based similarity
        similarities = []
        for i, gen_embed in enumerate(plot_embeddings):
            gen_context, gen_context_embed = get_context_window(
                generated_sentences,
                plot_embeddings,
                i
            )
            
            max_sim = -1
            best_context = ""
            best_index = -1
            
            for j in range(len(item['embeddings'])):
                orig_context, orig_context_embed = get_context_window(
                    item['original_sentences'],
                    item['embeddings'],
                    j
                )
                
                sim = cosine_similarity(
                    gen_context_embed.reshape(1, -1),
                    orig_context_embed.reshape(1, -1)
                )[0][0]
                
                if sim > max_sim:
                    max_sim = sim
                    best_context = orig_context
                    best_index = j
            
            similarities.append({
                'generated_context': gen_context,
                'original_context': best_context,
                'similarity_score': max_sim,
                'original_index': best_index
            })
        
        # N-gram overlap analysis
        ngram_overlap = calculate_ngram_overlap(generated_plot, ' '.join(item['original_sentences']))
        
        # Style metrics comparison
        generated_style = calculate_style_metrics(generated_plot)
        original_style = calculate_style_metrics(' '.join(item['original_sentences']))
        style_differences = compare_style_metrics(generated_style, original_style)
        
        analyzed_item = item.copy()
        analyzed_item.update({
            'embedding_similarities': similarities,
            'ngram_overlap': ngram_overlap,
            'style_metrics': {
                'generated': generated_style,
                'original': original_style,
                'differences': style_differences
            }
        })
        analyzed_data.append(analyzed_item)
    
    return analyzed_data

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

def generate_plot(author):
    sys.stderr.write(f"Generating plot for author: {author}\n")
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    prompt = f"Write a chapter of a new book that {author} could have written"
    
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": f"You are {author}."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        n=1,
        temperature=0.7,
    )
    
    plot = response.choices[0].message.content.strip()
    sys.stderr.write(f"Plot generated: {plot}\n")
    return plot

def generate_and_embed_plot(author, sentence_model, device):
    """Generate and embed plot with proper tensor conversion."""
    sys.stderr.write(f"Generating and embedding plot for author: {author}\n")
    plot = generate_plot(author)
    sentences = split_into_sentences(plot)
    set_seeds()
    
    # Get embeddings and convert to tensor
    plot_embeddings = sentence_model.encode(sentences, device=device)
    if isinstance(plot_embeddings, np.ndarray):
        plot_embeddings = torch.from_numpy(plot_embeddings)
    
    return plot, plot_embeddings

def visualize_results(analyzed_data, output_dir):
    """Create interactive visualizations of similarity analysis results using Plotly."""
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import numpy as np
    
    # 1. Embedding similarity distribution
    fig_dist = go.Figure()
    
    for item in analyzed_data:
        scores = [sim['similarity_score'] for sim in item['embedding_similarities']]
        fig_dist.add_trace(go.Histogram(
            x=scores,
            name=item['title'],
            opacity=0.7,
            nbinsx=20,
            hovertemplate=(
                "Similarity Score: %{x:.3f}<br>" +
                "Count: %{y}<br>" +
                "<extra></extra>"
            )
        ))
    
    fig_dist.update_layout(
        title="Distribution of Embedding Similarities",
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        barmode='overlay',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
        ),
        margin=dict(r=300),
        height=600,
        width=1000,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig_dist.write_html(os.path.join(output_dir, "embedding_similarities.html"))
    
    # 2. Style metrics comparison
    metrics = list(analyzed_data[0]['style_metrics']['differences'].keys())
    fig_box = go.Figure()
    
    for metric in metrics:
        values = [item['style_metrics']['differences'][metric] for item in analyzed_data]
        fig_box.add_trace(go.Box(
            y=values,
            name=metric,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate=(
                "Metric: %{x}<br>" +
                "Value: %{y:.3f}<br>" +
                "<extra></extra>"
            )
        ))
    
    fig_box.update_layout(
        title="Style Metric Differences",
        yaxis_title="Difference Value",
        template='plotly_white',
        height=600,
        width=1000,
        showlegend=False,
        boxmode='group'
    )
    
    fig_box.write_html(os.path.join(output_dir, "style_metrics.html"))
    
    # 3. Fixed heatmap of similarity scores
    # Prepare data for heatmap
    titles = [item['title'] for item in analyzed_data]
    max_length = max(len(item['embedding_similarities']) for item in analyzed_data)
    
    # Create a matrix of similarity scores
    similarity_matrix = np.zeros((len(analyzed_data), max_length))
    similarity_matrix.fill(np.nan)  # Fill with NaN initially
    
    for i, item in enumerate(analyzed_data):
        similarities = [sim['similarity_score'] for sim in item['embedding_similarities']]
        similarity_matrix[i, :len(similarities)] = similarities
    
    # Create heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        y=titles,
        colorscale='Viridis',
        connectgaps=False,  # Don't connect gaps in data
        hoverongaps=False,
        hovertemplate=(
            "Position: %{x}<br>" +
            "Title: %{y}<br>" +
            "Similarity: %{z:.3f}<br>" +
            "<extra></extra>"
        )
    ))
    
    fig_heat.update_layout(
        title="Similarity Scores Heatmap",
        xaxis_title="Position in Generated Text",
        yaxis_title="Original Text",
        height=max(400, len(analyzed_data) * 50),
        width=1000,
        template='plotly_white',
        xaxis=dict(
            tickmode='array',
            ticktext=[f'{i*10}%' for i in range(11)],
            tickvals=[i * (max_length-1)/10 for i in range(11)]
        )
    )
    
    fig_heat.write_html(os.path.join(output_dir, "similarity_heatmap.html"))
    
    # Create index.html
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Analysis Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .viz-link { 
                display: block; 
                margin: 10px 0; 
                padding: 10px; 
                background-color: #f0f0f0;
                text-decoration: none;
                color: #333;
                border-radius: 5px;
            }
            .viz-link:hover { background-color: #e0e0e0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Text Analysis Visualizations</h1>
            <a class="viz-link" href="embedding_similarities.html">Embedding Similarities Distribution</a>
            <a class="viz-link" href="style_metrics.html">Style Metrics Comparison</a>
            <a class="viz-link" href="similarity_heatmap.html">Similarity Scores Heatmap</a>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(index_html)

def generate_report(analyzed_data, output_dir):
    """Generate a detailed analysis report including cross-book comparisons."""
    report = "Text Similarity Analysis Report\n" + "="*80 + "\n\n"
    
    # First, collect all similarities across all books
    all_similarities = []
    for item in analyzed_data:
        for sim in item['embedding_similarities']:
            all_similarities.append({
                'book_title': item['title'],
                'similarity_score': sim['similarity_score'],
                'generated_context': sim['generated_context'],
                'original_context': sim['original_context']
            })
    
    # Print top 5 similarities across all books
    report += "Top 5 Most Similar Passages Across All Books\n" + "-"*80 + "\n"
    top_cross_book = sorted(all_similarities, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    for i, sim in enumerate(top_cross_book, 1):
        report += f"\n{i}. Similarity Score: {sim['similarity_score']:.4f}\n"
        report += f"From Book: {sim['book_title']}\n"
        report += f"Generated: {sim['generated_context']}\n"
        report += f"Original: {sim['original_context']}\n"
        report += "-"*40 + "\n"
    
    report += "\n" + "="*80 + "\n\n"
    
    # Then proceed with per-book analysis
    for item in analyzed_data:
        report += f"Analysis for: {item['title']}\n" + "-"*80 + "\n"
        
        # Overall statistics
        embedding_scores = [sim['similarity_score'] for sim in item['embedding_similarities']]
        report += f"\nEmbedding Similarity Statistics:\n"
        report += f"Average similarity: {np.mean(embedding_scores):.4f}\n"
        report += f"Max similarity: {np.max(embedding_scores):.4f}\n"
        report += f"N-gram overlap: {item['ngram_overlap']:.4f}\n\n"
        
        # Style metrics
        report += "Style Metrics Comparison:\n"
        for metric, diff in item['style_metrics']['differences'].items():
            report += f"{metric}: {diff:.4f} difference\n"
        
        # Most similar passages for this book
        report += "\nMost Similar Passages in This Book:\n"
        top_similarities = sorted(
            item['embedding_similarities'],
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:5]
        
        for i, sim in enumerate(top_similarities, 1):
            report += f"\n{i}. Similarity Score: {sim['similarity_score']:.4f}\n"
            report += f"Generated: {sim['generated_context']}\n"
            report += f"Original: {sim['original_context']}\n"
            report += "-"*40 + "\n"
        
        report += "\n" + "="*80 + "\n"
    
    # Add summary statistics across all books
    all_scores = [sim['similarity_score'] for sim in all_similarities]
    report += "\nOverall Statistics Across All Books\n" + "-"*80 + "\n"
    report += f"Average similarity across all books: {np.mean(all_scores):.4f}\n"
    report += f"Maximum similarity found: {np.max(all_scores):.4f}\n"
    report += f"Minimum similarity found: {np.min(all_scores):.4f}\n"
    report += f"Standard deviation of similarities: {np.std(all_scores):.4f}\n"
    
    with open(os.path.join(output_dir, "analysis_report.txt"), 'w', encoding='utf-8') as f:
        f.write(report)

def main(author, api_key):
    """
    Main function with improved analysis pipeline.
    """
    sys.stderr.write(f"Starting enhanced analysis for author: {author}\n")

    set_seeds()
    
    # Setup remains the same
    metadata_file = "metadata/metadata.csv"
    raw_dir = "data/raw"
    output_dir = "output"
    ensure_dir(output_dir)
    
    device = get_device()
    sys.stderr.write(f"Using device: {device}\n")
    
    # Load models and data
    sys.stderr.write("Loading models...\n")
    sentence_model = SentenceTransformer('all-mpnet-base-v2').to(device)
    
    metadata = load_metadata(metadata_file)
    author_metadata = metadata[metadata['author'] == author]
    
    if author_metadata.empty:
        sys.stderr.write(f"No books found for author: {author}\n")
        return
    
    cache_dir = get_cache_path(output_dir)

    # Process and embed texts
    embedded_data = process_and_embed_texts(author_metadata, raw_dir, sentence_model, device, cache_dir)
    
    # Generate and embed new plot
    generated_plot, plot_embeddings = generate_and_embed_plot(author, sentence_model, device)
    
    # Perform enhanced analysis
    sys.stderr.write("Performing contextual similarity analysis...\n")
    analyzed_data = analyze_similarity(embedded_data, generated_plot, plot_embeddings)
    visualize_results(analyzed_data, output_dir)
    generate_report(analyzed_data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze texts by a specific author.")
    parser.add_argument("author", type=str, help="Name of the author to analyze")
    parser.add_argument("api_key", type=str, help="OpenAI API key", nargs='?', default=os.getenv("OPENAI_API_KEY"))
    
    args = parser.parse_args()
    main(args.author, args.api_key)

