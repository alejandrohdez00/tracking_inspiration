import numpy as np
import matplotlib.pyplot as plt
import os

def compute_similarity(embed1, embed2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

def analyze_similarity(embedded_data, generated_plot, plot_embeddings):
    """Analyze similarity between generated plot and original sentences."""
    all_similarities = []
    all_most_similar_sentences = []
    all_most_similar_books = []

    for gen_embed in plot_embeddings:
        max_sim = -1
        max_sim_sentence = ""
        max_sim_book = ""
        for item in embedded_data:
            for i, orig_embed in enumerate(item['embeddings']):
                sim = compute_similarity(gen_embed, orig_embed)
                if sim > max_sim:
                    max_sim = sim
                    max_sim_sentence = item['original_sentences'][i]
                    max_sim_book = item['title']
        all_similarities.append(max_sim)
        all_most_similar_sentences.append(max_sim_sentence)
        all_most_similar_books.append(max_sim_book)

    return {
        'generated_plot': generated_plot,
        'similarities': all_similarities,
        'most_similar_sentences': all_most_similar_sentences,
        'most_similar_books': all_most_similar_books
    }

def visualize_similarity(data, output_dir):
    """Create visualization of sentence similarities."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(data['similarities'])), data['similarities'])
    plt.title(f"Sentence Similarities for Generated Plot")
    plt.xlabel("Generated Sentence Index")
    plt.ylabel("Similarity to Original Books")
    plt.savefig(os.path.join(output_dir, "generated_plot_similarity.png"))
    plt.close()

def generate_report(data, generated_plot, output_dir):
    """Generate a summary report of the findings."""
    report = "Text Analysis and Generation Report\n\n"
    
    report += f"Generated Plot:\n{generated_plot}\n\n"
    
    report += "Sentence-by-Sentence Analysis:\n\n"
    for i, (gen_sentence, orig_sentence, similarity, book) in enumerate(zip(
        data['generated_plot'].split('.'),
        data['most_similar_sentences'],
        data['similarities'],
        data['most_similar_books']
    )):
        if gen_sentence.strip():  # Skip empty sentences
            report += f"Generated: {gen_sentence.strip()}\n"
            report += f"Most similar to: \"{orig_sentence}\"\n"
            report += f"Similarity: {similarity:.4f}\n"
            report += f"Book: {book}\n\n"
    
    report += f"Average Similarity: {np.mean(data['similarities']):.4f}\n"
    report += f"Max Similarity: {np.max(data['similarities']):.4f}\n"
    
    with open(os.path.join(output_dir, "report.txt"), 'w') as f:
        f.write(report)
