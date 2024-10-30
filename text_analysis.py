"""Main script coordinating the text analysis pipeline"""
import sys
import os
import argparse
from sentence_transformers import SentenceTransformer
import pandas as pd
from text_processing import process_raw_text, split_into_sentences
from embedding_handler import process_and_embed_texts, get_context_window
from style_analysis import calculate_style_metrics, calculate_ngram_overlap, compare_style_metrics
from plot_generation import generate_plot, generate_and_embed_plot
from visualization import visualize_results, create_position_similarity_arcs
from reporting import generate_report
from similarity_analysis import analyze_similarity
from utils import get_device, ensure_dir, set_seeds
from cache_handling import get_cache_path

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
    
    metadata = pd.read_csv(metadata_file)
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
    create_position_similarity_arcs(analyzed_data, output_dir)
    generate_report(analyzed_data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze texts by a specific author.")
    parser.add_argument("author", type=str, help="Name of the author to analyze")
    parser.add_argument("api_key", type=str, help="OpenAI API key", nargs='?', default=os.getenv("OPENAI_API_KEY"))
    
    args = parser.parse_args()
    main(args.author, args.api_key)

