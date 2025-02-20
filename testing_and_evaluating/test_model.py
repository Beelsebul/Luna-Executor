import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import tkinter as tk
import os
from tkinter import ttk

# Specifying paths
base_dir = os.path.join(os.path.dirname(__file__), '..')
model_checkpoint = os.path.join(base_dir, 'minihex_red', 'checkpoint-319150')
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

# Function to predict next tokens
def predict_next_tokens(input_text, top_k=5):
    # Adding <|startoftext|> at the beginning if the text is not empty
    if not input_text:
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
    else:
        # Adding <|startoftext|> before the input text
        input_text = '<|startoftext|>' + input_text
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    
    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
    
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices[0])
    top_k_probs = top_k_probs[0].tolist()
    
    return top_k_tokens, top_k_probs

# Function to update predictions
def update_predictions(*args):
    input_text = input_entry.get()
    tokens, probs = predict_next_tokens(input_text)
    
    for i, (token, prob) in enumerate(zip(tokens, probs)):
        prediction_labels[i].config(text=f"{token}: {prob:.4f}")

# Setting up the interface
root = tk.Tk()
root.title("5 Possible Moves")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

input_label = ttk.Label(mainframe, text="Input:")
input_label.grid(column=0, row=0, sticky=tk.W)

input_entry = ttk.Entry(mainframe, width=50)
input_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))
input_entry.bind("<KeyRelease>", update_predictions)

predictions_frame = ttk.Frame(mainframe, padding="10 10 10 10")
predictions_frame.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))

prediction_labels = []
for i in range(5):
    label = ttk.Label(predictions_frame, text="", width=50, anchor="w")
    label.grid(column=0, row=i, sticky=(tk.W, tk.E))
    prediction_labels.append(label)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
mainframe.columnconfigure(1, weight=1)

root.mainloop()
