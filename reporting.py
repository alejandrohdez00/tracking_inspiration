"""Handle report generation"""
import numpy as np
import os

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