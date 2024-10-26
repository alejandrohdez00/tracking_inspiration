import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import torch
from utils import get_device, ensure_dir
from analysis import compute_similarity
import argparse
import sys
import gc
from openai import OpenAI
import re

# Function definitions will go here

def load_metadata(file_path):
    """Load metadata from CSV file."""
    return pd.read_csv(file_path)

def process_raw_text(file_path):
    """Process raw text file, removing non-book content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Remove header and footer (adjust as needed)
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

def process_and_embed_texts(metadata, raw_dir, model, device):
    sys.stderr.write("Starting process_and_embed_texts...\n")
    embedded_data = []
    for _, row in metadata.iterrows():
        sys.stderr.write(f"Processing book: {row['title']}\n")
        file_path = os.path.join(raw_dir, f"{row['id']}_raw.txt")
        if os.path.exists(file_path):
            text = process_raw_text(file_path)
            sentences = split_into_sentences(text)
            sys.stderr.write(f"Embedding {len(sentences)} sentences...\n")
            embeddings = model.encode(sentences, device=device)
            embedded_data.append({
                'id': row['id'],
                'title': row['title'],
                'author': row['author'],
                'embeddings': embeddings,
                'original_sentences': sentences
            })
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
    sys.stderr.write(f"Generating and embedding plot for author: {author}\n")
    plot = generate_plot(author)  # Removed api_key parameter since it's not used in generate_plot
    # Use the new split_into_sentences function instead of simple split
    sentences = split_into_sentences(plot)
    plot_embeddings = sentence_model.encode(sentences, device=device)
    return plot, plot_embeddings


def analyze_similarity(data, generated_plot, plot_embeddings):
    """Analyze similarity between generated and original sentences."""
    analyzed_data = []
    for item in data:
        similarities = []
        for gen_embed in plot_embeddings:
            sim = max(compute_similarity(gen_embed, orig_embed) for orig_embed in item['embeddings'])
            similarities.append(sim)
        analyzed_item = item.copy()
        analyzed_item['similarities'] = similarities
        analyzed_data.append(analyzed_item)
    return analyzed_data

def visualize_similarity(data, output_dir):
    """Create visualization of sentence similarities."""
    for item in data:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(item['similarities'])), item['similarities'])
        plt.title(f"Sentence Similarities for '{item['title']}' by {item['author']}")
        plt.xlabel("Sentence Index")
        plt.ylabel("Similarity to Generated Plot")
        plt.savefig(os.path.join(output_dir, f"{item['id']}_similarity.png"))
        plt.close()


def generate_report(data, generated_plot, plot_embeddings, output_dir):
    """Generate a summary report of the findings."""
    report = "Text Analysis and Generation Report\n\n"
    generated_sentences = [sent.strip() for sent in generated_plot.split('.') if sent.strip()]
    
    # Add validation at the start of report generation
    report += f"Debug Information:\n"
    report += f"Number of generated sentences: {len(generated_sentences)}\n"
    report += f"Number of plot embeddings: {len(plot_embeddings)}\n"
    report += f"First embedding shape: {plot_embeddings[0].shape}\n\n"
    
    for i, gen_sentence in enumerate(generated_sentences):
        if i >= len(plot_embeddings):
            report += f"Warning: More sentences than embeddings. Stopping at {i}\n"
            break
            
        report += f"Generated Sentence {i+1}: {gen_sentence}\n"
        
        max_similarity = -1
        most_similar_sentence = ""
        most_similar_book = ""
        
        for item in data:
            for j, orig_sentence in enumerate(item['original_sentences']):
                try:
                    sim = compute_similarity(plot_embeddings[i], item['embeddings'][j])
                    if sim > max_similarity:
                        max_similarity = sim
                        most_similar_sentence = orig_sentence
                        most_similar_book = item['title']
                except Exception as e:
                    report += f"Error computing similarity for sentence {i}, original sentence {j}: {str(e)}\n"
        
        report += f"Most Similar Sentence: \"{most_similar_sentence}\"\n"
        report += f"Similarity: {max_similarity:.4f}\n"
        report += f"Book: {most_similar_book}\n\n"
    
    with open(os.path.join(output_dir, "report.txt"), 'w') as f:
        f.write(report)

def main(author, api_key):
    sys.stderr.write(f"Starting analysis for author: {author}\n")
    
    # Set up directories and models
    metadata_file = "metadata/metadata.csv"
    raw_dir = "data/raw"
    output_dir = "output"
    ensure_dir(output_dir)

    device = get_device()
    sys.stderr.write(f"Using device: {device}\n")

    sys.stderr.write("Loading models...\n")
    sentence_model = SentenceTransformer('all-mpnet-base-v2').to(device)
    sys.stderr.write("Models loaded.\n")

    # Load and process data
    sys.stderr.write("Loading metadata...\n")
    metadata = load_metadata(metadata_file)
    author_metadata = metadata[metadata['author'] == author]
    
    if author_metadata.empty:
        sys.stderr.write(f"No books found for author: {author}\n")
        return

    sys.stderr.write(f"Found {len(author_metadata)} books for {author}.\n")

    embedded_data = process_and_embed_texts(author_metadata, raw_dir, sentence_model, device)
    
    # Clear CUDA memory after embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Store both plot and embeddings
    generated_plot, plot_embeddings = generate_and_embed_plot(author, sentence_model, device)
    
    # Add some validation prints
    sys.stderr.write(f"Generated plot sentences: {len(generated_plot.split('.'))}\n")
    sys.stderr.write(f"Plot embeddings: {len(plot_embeddings)}\n")
    
    # Clear CUDA memory after plot generation and embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    sys.stderr.write("Analyzing similarity...\n")
    analyzed_data = analyze_similarity(embedded_data, generated_plot, plot_embeddings)

    sys.stderr.write("Visualizing similarity...\n")
    visualize_similarity(analyzed_data, output_dir)
    
    sys.stderr.write("Generating report...\n")
    # Pass plot_embeddings to generate_report
    generate_report(analyzed_data, generated_plot, plot_embeddings, output_dir)

    sys.stderr.write(f"Analysis complete for author: {author}. Results saved in the 'output' directory.\n")

    # Final CUDA memory clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze texts by a specific author.")
    parser.add_argument("author", type=str, help="Name of the author to analyze")
    parser.add_argument("api_key", type=str, help="OpenAI API key", nargs='?', default=os.getenv("OPENAI_API_KEY"))    

    args = parser.parse_args()

    main(args.author, args.api_key)
