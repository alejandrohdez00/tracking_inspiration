"""Handle plot generation using GPT"""
import os
from openai import OpenAI
import sys
import torch
import numpy as np
from utils import set_seeds
from text_processing import split_into_sentences



def generate_plot(author):
    sys.stderr.write(f"Generating plot for author: {author}\n")
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    prompt = f"Write a plot for of a new book that {author} could have written"
    
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
