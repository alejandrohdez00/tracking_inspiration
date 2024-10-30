"""Handle similarity analysis operations"""
from sklearn.metrics.pairwise import cosine_similarity
import sys
from text_processing import split_into_sentences
from style_analysis import calculate_style_metrics, calculate_ngram_overlap, compare_style_metrics
from embedding_handler import get_context_window


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