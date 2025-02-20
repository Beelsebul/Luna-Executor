from flask import Flask, request, render_template, jsonify
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import os

app = Flask(__name__)

# Define paths for each checkpoint
base_path = os.path.join(os.path.dirname(__file__), '..')
red_model_checkpoint = os.path.join(base_path, 'mini_red_test', 'checkpoint-765960')
blue_model_checkpoint = os.path.join(base_path, 'mini_blue_test', 'checkpoint-785460')
tokenizer_path = os.path.join(base_path, 'tokenizer')

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = '<|pad|>'
tokenizer.bos_token = '<|startoftext|>'
tokenizer.eos_token = '<|endoftext|>'
tokenizer.unk_token = '<|unk|>'

# Load models for red and blue players
red_model = GPT2LMHeadModel.from_pretrained(red_model_checkpoint)
blue_model = GPT2LMHeadModel.from_pretrained(blue_model_checkpoint)

# Ensure models have resized token embeddings to match tokenizer
red_model.resize_token_embeddings(len(tokenizer))
blue_model.resize_token_embeddings(len(tokenizer))

# Prediction function for generating next moves
def predict_next_tokens(model, input_text, top_k=140):
    if not input_text:
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
    else:
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

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# AI move generation route
@app.route('/next-move', methods=['POST'])
def next_move():
    data = request.get_json()
    moves = data.get('moves', '')
    
    # Determine which player's turn it is based on move count
    current_turn = len(moves.split()) % 2
    model = red_model if current_turn == 0 else blue_model  # Red plays first, then Blue

    # Generate the next moves
    top_moves, probs = predict_next_tokens(model, moves)

    return jsonify({'next_moves': top_moves, 'probabilities': probs})

if __name__ == '__main__':
    app.run(debug=True)
