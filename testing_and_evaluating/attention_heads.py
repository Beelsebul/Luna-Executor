import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import tkinter as tk
from tkinter import ttk
import os
import matplotlib.pyplot as plt
import numpy as np

# Specifying paths
base_dir = os.path.join(os.path.dirname(__file__), '..')
model_checkpoint = os.path.join(base_dir, 'minihex_red', 'checkpoint-765960_ft_r_gen_5') 
tokenizer_path = os.path.join(base_dir, "tokenizer")

# Loading tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

# Ensuring special tokens are set
tokenizer.pad_token = '<|pad|>'
tokenizer.bos_token = '<|startoftext|>'
tokenizer.eos_token = '<|endoftext|>'
tokenizer.unk_token = '<|unk|>'

# Updating model embeddings to match the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Setting special tokens in model configuration
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.unk_token_id = tokenizer.unk_token_id

# Global variables to store attention weights and input tokens
attention_weights = None
input_tokens = None

# Function to predict with attention
def predict_with_attention(input_text):
    global attention_weights, input_tokens
    if not input_text:
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
    else:
        input_text = '<|startoftext|>' + input_text
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    attention_weights = outputs.attentions
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Function to visualize attention for a specific head and layer
def visualize_attention(layer, head):
    if attention_weights is None or input_tokens is None:
        return

    attention = attention_weights[layer][0, head].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(attention, cmap='viridis')
    plt.colorbar(cax)

    ax.set_xticks(range(len(input_tokens)))
    ax.set_yticks(range(len(input_tokens)))
    ax.set_xticklabels(input_tokens, rotation=90)
    ax.set_yticklabels(input_tokens)

    for (i, j), val in np.ndenumerate(attention):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val > 0.5 else 'black')

    plt.title(f"Layer {layer + 1}, Head {head + 1}")
    plt.show()

# Function to update predictions and show buttons for attention heads
def update_predictions(*args):
    input_text = input_entry.get()
    predict_with_attention(input_text)

    for widget in button_frame.winfo_children():
        widget.destroy()

    if attention_weights:
        num_layers = len(attention_weights)
        num_heads = attention_weights[0].size(1)

        for layer in range(num_layers):
            layer_label = ttk.Label(button_frame, text=f"Layer {layer + 1}")
            layer_label.grid(column=layer, row=0, padx=5, pady=5)

            for head in range(num_heads):
                button = ttk.Button(
                    button_frame,
                    text=f"Head {head + 1}",
                    command=lambda l=layer, h=head: visualize_attention(l, h)
                )
                button.grid(column=layer, row=head + 1, padx=5, pady=5)

# Setting up the interface
root = tk.Tk()
root.title("Attention Visualizer")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

input_label = ttk.Label(mainframe, text="Input:")
input_label.grid(column=0, row=0, sticky=tk.W)

input_entry = ttk.Entry(mainframe, width=50)
input_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))
input_entry.bind("<KeyRelease>", update_predictions)

button_frame = ttk.Frame(mainframe, padding="10 10 10 10")
button_frame.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
mainframe.columnconfigure(1, weight=1)

root.mainloop()
