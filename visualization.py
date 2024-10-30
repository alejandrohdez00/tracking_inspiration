"""Handle all visualization related functions"""
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import os

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
    
    # 4. New line plot for top 10 most similar books
    # Calculate mean similarity for each book
    book_mean_similarities = []
    for item in analyzed_data:
        mean_sim = np.mean([sim['similarity_score'] for sim in item['embedding_similarities']])
        book_mean_similarities.append({
            'title': item['title'],
            'mean_similarity': mean_sim,
            'similarities': [sim['similarity_score'] for sim in item['embedding_similarities']]
        })
    
    # Sort and get top 10 books
    top_10_books = sorted(book_mean_similarities, 
                         key=lambda x: x['mean_similarity'], 
                         reverse=True)[:10]
    
    # Create line plot
    fig_lines = go.Figure()
    
    # Define a color scale for 10 distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, book in enumerate(top_10_books):
        # Create x-axis positions as percentages
        x_positions = np.linspace(0, 100, len(book['similarities']))
        
        fig_lines.add_trace(go.Scatter(
            x=x_positions,
            y=book['similarities'],
            mode='lines',
            name=f"{book['title']} (mean: {book['mean_similarity']:.3f})",
            line=dict(color=colors[idx], width=2),
            hovertemplate=(
                "Position: %{x:.1f}%<br>" +
                "Similarity: %{y:.3f}<br>" +
                "<extra></extra>"
            )
        ))
    
    fig_lines.update_layout(
        title="Similarity Scores Across Generated Text for Top 10 Most Similar Books",
        xaxis_title="Position in Generated Text (%)",
        yaxis_title="Similarity Score",
        height=600,
        width=1000,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        margin=dict(b=150),  # Increase bottom margin for legend
        hovermode='x unified'
    )
    
    fig_lines.update_xaxes(range=[0, 100])
    fig_lines.update_yaxes(range=[0, 1])
    
    fig_lines.write_html(os.path.join(output_dir, "top_books_similarity_lines.html"))
    
    # Update index.html to include the new visualization
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
            <a class="viz-link" href="top_books_similarity_lines.html">Top 10 Books Similarity Lines</a>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(index_html)


def create_position_similarity_arcs(analyzed_data, output_dir):
    """Create an interactive visualization showing position relationships between texts."""
    import plotly.graph_objects as go
    import numpy as np
    import os
    
    # Get top 5 books (instead of 10 for better visibility)
    book_mean_similarities = []
    for item in analyzed_data:
        mean_sim = np.mean([sim['similarity_score'] for sim in item['embedding_similarities']])
        book_mean_similarities.append({
            'title': item['title'],
            'mean_similarity': mean_sim,
            'similarities': item['embedding_similarities']
        })
    
    top_5_books = sorted(book_mean_similarities, 
                        key=lambda x: x['mean_similarity'], 
                        reverse=True)[:5]
    
    # Create subplots - one for each book
    fig = go.Figure()
    
    # Define colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create a subplot for each book
    for book_idx, book in enumerate(top_5_books):
        # Create a 2D heatmap-style visualization
        similarities = []
        gen_positions = []
        orig_positions = []
        
        # Collect all similarity pairs
        for sim in book['similarities']:
            if sim['similarity_score'] > 0.5:  # Only show stronger connections
                gen_pos = sim['original_index'] / len(book['similarities']) * 100
                orig_pos = sim['original_index'] / len(book['similarities']) * 100
                similarities.append(sim['similarity_score'])
                gen_positions.append(gen_pos)
                orig_positions.append(orig_pos)
        
        # Add scatter plot for this book
        fig.add_trace(go.Scatter(
            x=gen_positions,
            y=orig_positions,
            mode='markers',
            marker=dict(
                size=10,
                color=similarities,
                colorscale='Viridis',
                showscale=True if book_idx == 0 else False,  # Only show colorbar for first book
                colorbar=dict(title="Similarity Score"),
                symbol='diamond',
            ),
            name=f"{book['title']} (mean: {book['mean_similarity']:.3f})",
            hovertemplate=(
                "Generated Text Position: %{x:.1f}%<br>" +
                "Original Text Position: %{y:.1f}%<br>" +
                "Similarity Score: %{marker.color:.3f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add diagonal line for reference
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title="Position Relationships Between Generated and Original Texts<br><sub>Points show where similar content appears in both texts. Diagonal line represents perfect position alignment.</sub>",
        xaxis_title="Position in Generated Text (%)",
        yaxis_title="Position in Original Text (%)",
        height=800,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        margin=dict(t=150, b=50),
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(240,240,240,0.5)',
        hovermode='closest'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     dtick=20, ticksuffix="%")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     dtick=20, ticksuffix="%")
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, "position_relationships.html"))
    
    # Update index.html
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
            <a class="viz-link" href="top_books_similarity_lines.html">Top 10 Books Similarity Lines</a>
            <a class="viz-link" href="position_relationships.html">Position Relationships</a>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(index_html)