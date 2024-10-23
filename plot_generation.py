from utils import get_device

def generate_plot(author, model, tokenizer):
    """Generate a plot in the style of the given author using GPT-2."""
    prompt = f"In the style of {author}, write a short plot for a new book:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(get_device())
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_and_embed_plots(embedded_data, gpt_model, gpt_tokenizer, sentence_model):
    """Generate and embed plots for each author."""
    for data in embedded_data:
        plot = generate_plot(data['author'], gpt_model, gpt_tokenizer)
        plot_embeddings = sentence_model.encode(plot.split('.'), device=get_device())
        data['generated_plot'] = plot
        data['generated_embeddings'] = plot_embeddings
    return embedded_data
