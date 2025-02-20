import numpy as np
import os
import re
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm 
import pandas as pd
from pandas import concat
import csv
import string
import time

def create_board():
    return np.zeros((11, 11), dtype=int)

def convert_move_to_coords(move):
    letter = move[0].upper()
    number = int(move[1:]) - 1
    x = ord(letter) - ord('A')
    y = number
    return x, y

def coords_to_move(x, y):
    letter = chr(x + ord('A'))
    number = y + 1
    return f"{letter}{number}"

def is_valid_move(move, size=11):
    # Check that move is valid within board boundaries
    if len(move) < 2:
        return False
    letter = move[0].upper()
    if not 'A' <= letter <= chr(ord('A') + size - 1):
        return False
    try:
        number = int(move[1:])
        return 1 <= number <= size
    except ValueError:
        return False

def apply_moves(board, moves):
    # Applying moves to the board
    applied_moves = []
    _allow_swap_rule = True
    for i, move in enumerate(moves):
        if not is_valid_move(move):
            return 6, board  # Invalid move

        x, y = convert_move_to_coords(move)

        if i == 1 and _allow_swap_rule:
            if move == moves[0]:  # If the second move matches the first
                board[y, x] = 0  # Remove the red stone
                y, x = x, y  # Swap coordinates
                swap_move = coords_to_move(x, y)
                board[y, x] = 2  # Place the blue stone
                applied_moves = [(x, y)]
            else:
                board[y, x] = 2  # Regular blue move
                applied_moves.append((x, y))
        else:
            if board[y, x] != 0:
                return 5, board  # Cell is occupied
            if i >= 2 and (x, y) in applied_moves:
                return 5, board  # Cell is occupied
            board[y, x] = 1 if i % 2 == 0 else 2
            applied_moves.append((x, y))

    return None, board

def parse_moves_sequence(sequence):
    return sequence.split()

def get_neighbors(i, j, size):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    return [(i + di, j + dj) for di, dj in directions if 0 <= i + di < size and 0 <= j + dj < size]

def dfs(board, start, player, goal_edge):
    # Depth-first search to check if there is a connected path
    size = len(board)
    stack = [start]
    visited = set()

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            i, j = current

            if goal_edge(i, j):
                return True

            for ni, nj in get_neighbors(i, j, size):
                if board[ni, nj] == player and (ni, nj) not in visited:
                    stack.append((ni, nj))

    return False

def check_winner_red(board):  # Red (1) connects top and bottom
    size = len(board)
    for j in range(size):
        if board[0, j] == 1:  # Check the top edge
            if dfs(board, (0, j), 1, lambda i, j: i == size - 1):
                return True
    return False

def check_winner_blue(board):  # Blue (2) connects left and right
    size = len(board)
    for i in range(size):
        if board[i, 0] == 2:  # Check the left edge
            if dfs(board, (i, 0), 2, lambda i, j: j == size - 1):
                return True
    return False

def check_winner(board):
    # Determine if there is a winner
    if check_winner_red(board):
        return 3
    elif check_winner_blue(board):
        return 4
    else:
        return 2

def play_game(moves_sequence):
    # Execute a full game and return the result
    board = create_board()
    moves = parse_moves_sequence(moves_sequence)
    error, board = apply_moves(board, moves)

    if error:
        return error
    winner = check_winner(board)
    return winner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_path = os.path.join(os.path.dirname(__file__), '..')
red_checkpoints_dir = os.path.join(base_path, "mini_red_test")
blue_checkpoints_dir = os.path.join(base_path, "mini_blue_test")
logs_dir = os.path.join(base_path, "logs")
all_games_df = pd.DataFrame(columns=['Red Checkpoint', 'Blue Checkpoint', 'Game Moves', 'Winner'])

# Function to filter and sort checkpoints
def filter_and_sort_checkpoints(checkpoints):
    # Filter checkpoints to include only those with 'checkpoint-'
    filtered = [cp for cp in checkpoints if 'checkpoint-' in cp]
    
    # Sort by number after 'checkpoint-'
    sorted_checkpoints = sorted(filtered, key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    
    return sorted_checkpoints

# List of checkpoint directories
red_checkpoints = [os.path.join(red_checkpoints_dir, cp) for cp in os.listdir(red_checkpoints_dir)]
blue_checkpoints = [os.path.join(blue_checkpoints_dir, cp) for cp in os.listdir(blue_checkpoints_dir)]

# Apply function to sort and filter checkpoints
sorted_red_checkpoints = filter_and_sort_checkpoints(red_checkpoints)
sorted_blue_checkpoints = filter_and_sort_checkpoints(blue_checkpoints)
sum_checkpoints = sorted_red_checkpoints + sorted_blue_checkpoints

# Total number of steps
total_games = len(red_checkpoints) * len(blue_checkpoints) * 121

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained('C:/Users/Timur/Downloads/hex_agony/tokenizer')

# Define all possible positions with whitespace
columns = list(string.ascii_uppercase[:11])  # A-K
rows = [str(i) for i in range(1, 12)]        # 1-11
positions = [f"{col}{row}" for col in columns for row in rows]  # ['A1', 'A2', ..., 'K11']

# Function to load a model
def load_model(checkpoint_path):
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model

def predict_next_tokens(model, moves_sequences, top_k=125):
    # Adding <|startoftext|> to each move sequence
    input_texts = ['<|startoftext|> ' + moves for moves in moves_sequences]  # Adding special token
    
    # Tokenization with padding and truncation
    inputs = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    
    # Get top-K predictions
    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
    
    top_k_tokens = []
    for i in range(len(input_texts)):
        tokens = [tokenizer.decode([token_id]).strip() for token_id in top_k_indices[i].tolist()]
        top_k_tokens.append(tokens)
    
    return top_k_tokens

# Creating a 3D cube for games
def shape_the_cube():
    games_cube = [[[None for _ in range(len(sorted_blue_checkpoints) * 121 + 1)] 
                    for _ in range(3)] 
                    for _ in range(len(sorted_red_checkpoints) + len(sorted_blue_checkpoints))]
    
    stats_cube = [[[0, 0] for _ in range(len(sorted_blue_checkpoints) + 1)]
                    for _ in range(len(sorted_red_checkpoints) +1)]
    
    stats_cube[0][0] = ["red_vertically", "blue_horizontally"]

    number_to_checkpoint = []
    # Adding values for each blue player
    for blue_player in range(len(sorted_blue_checkpoints)):
        # Adding red checkpoints for the blue player
        games_cube[blue_player][0][0] = blue_player
        games_cube[blue_player][1][0] = 2  # Move number
        games_cube[blue_player][2][0] = len(sorted_red_checkpoints) * 121  # Number of games in the list
        number_to_checkpoint.append(sorted_blue_checkpoints[blue_player])
        stats_cube[0][blue_player + 1] = sorted_blue_checkpoints[blue_player]
        # Adding 121 moves for each red checkpoint
        for red_checkpoint in range(len(sorted_red_checkpoints)):
            for idx in range(len(positions)):
                index = 1 + red_checkpoint * 121 + idx
                games_cube[blue_player][0][index] = red_checkpoint + len(sorted_blue_checkpoints)
                games_cube[blue_player][1][index] = positions[idx]
    for red_player in range(len(sorted_red_checkpoints)):
        games_cube[red_player + len(sorted_blue_checkpoints)][0][0] = red_player + len(sorted_blue_checkpoints)
        games_cube[red_player + len(sorted_blue_checkpoints)][1][0] = 3  # Move number
        games_cube[red_player + len(sorted_blue_checkpoints)][2][0] = 0  # Number of games in the list
        number_to_checkpoint.append(sorted_red_checkpoints[red_player])
        stats_cube[red_player + 1][0] = sorted_red_checkpoints[red_player]
    return games_cube, number_to_checkpoint, stats_cube

games_cube, number_to_checkpoint, stats_cube = shape_the_cube()

def count_remaining_games():
    total = 0 
    for i in range(len(games_cube)):
        total += games_cube[i][2][0]
    return total

def update_checkpoint_stats(map, games_map, main_checkpoint, opponent_checkpoint, game_to_check, result):
    red_checkpoint, blue_checkpoint = (main_checkpoint, opponent_checkpoint) if main_checkpoint > opponent_checkpoint else (opponent_checkpoint, main_checkpoint)
    if result == 3:
        map[red_checkpoint - len(sorted_blue_checkpoints) + 1][blue_checkpoint + 1][0] += 1
        games_map = pd.concat([games_map, pd.DataFrame([{
                        'Red Checkpoint': red_checkpoint,
                        'Blue Checkpoint': blue_checkpoint,
                        'Game Moves': game_to_check,
                        'Winner': 'Red'
                    }])], ignore_index=True)
    else:
        map[red_checkpoint - len(sorted_blue_checkpoints) + 1][blue_checkpoint + 1][1] += 1
        games_map = pd.concat([games_map, pd.DataFrame([{
                        'Red Checkpoint': red_checkpoint,
                        'Blue Checkpoint': blue_checkpoint,
                        'Game Moves': game_to_check,
                        'Winner': 'Blue'
                    }])], ignore_index=True)
    return map, games_map

def preload_models():
    models = []
    # Load models
    for i in range(len(games_cube)):
        models.append(load_model(number_to_checkpoint[i]))
    return models

def main():
    global all_games_df, stats_cube
    move_number = 1
    games_left = total_games
    start_time = time.time()
    models = preload_models()
    pbar = tqdm(total=total_games, desc="Processing Games", unit="game")
    while games_left != 0:
        move_number += 2
        games_left = count_remaining_games()
        for checkpoint_to_play in range(len(games_cube)):
            model = models[checkpoint_to_play]
            # Collect move sequences
            move_sequences = []
            num_games = games_cube[checkpoint_to_play][2][0]
            game_numbers = []
            for game_number in range(1, num_games + 1):
                move_sequences.append(games_cube[checkpoint_to_play][1][game_number])
                game_numbers.append(game_number)
            # Predict next tokens in batch
            if move_sequences:
                all_top_k_tokens = predict_next_tokens(model, move_sequences)
                # Assign predictions back
                for idx, game_number in enumerate(game_numbers):
                    games_cube[checkpoint_to_play][2][game_number] = all_top_k_tokens[idx]
            # Process each game
            for idx, game_number in enumerate(game_numbers):
                game_result = 0
                move_index = 0
                while game_result < 2 or game_result > 4:
                    if move_index >= len(games_cube[checkpoint_to_play][2][game_number]):
                        break  # No more moves to try
                    next_move = games_cube[checkpoint_to_play][2][game_number][move_index]
                    game_to_check = f"{games_cube[checkpoint_to_play][1][game_number]} {next_move}"
                    game_result = play_game(game_to_check)
                    move_index += 1
                games_cube[checkpoint_to_play][1][game_number] = game_to_check
                if game_result == 2:
                    opponent = games_cube[checkpoint_to_play][0][game_number]
                    games_cube[opponent][2][0] += 1
                    new_index = games_cube[opponent][2][0]
                    games_cube[opponent][1][new_index] = game_to_check
                    games_cube[opponent][0][new_index] = games_cube[checkpoint_to_play][0][0]
                else:  # Game over, update stats
                    stats_cube, all_games_df = update_checkpoint_stats(stats_cube, all_games_df, checkpoint_to_play, games_cube[checkpoint_to_play][0][game_number], game_to_check, game_result)
            pbar.update(total_games - games_left - pbar.n)  
            pbar.set_postfix(move=move_number)
            games_cube[checkpoint_to_play][2][0] = 0    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n Execution time: {execution_time} sec")
    # Close progress bar upon completion
    pbar.close()
    all_games_df.to_csv(os.path.join(logs_dir, 'all_games_results.csv'), index=False, sep=';')
    with open(os.path.join(logs_dir, 'games_matrix.csv'), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(stats_cube)

if __name__ == "__main__":
    main()
