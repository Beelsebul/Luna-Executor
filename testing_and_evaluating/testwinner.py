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
    # Creating an empty 11x11 game board
    return np.zeros((11, 11), dtype=int)

def convert_move_to_coords(move):
    # Converting a move notation (e.g., "A1") to board coordinates (x, y)
    letter = move[0].upper()
    number = int(move[1:]) - 1
    x = ord(letter) - ord('A')
    y = number
    return x, y

def coords_to_move(x, y):
    # Converting board coordinates (x, y) back to a move notation (e.g., "A1")
    letter = chr(x + ord('A'))
    number = y + 1
    return f"{letter}{number}"

def is_valid_move(move, size=11):
    # Checking if the given move is valid within the board size and format
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
    # Applying a sequence of moves to the board
    applied_moves = []
    _allow_swap_rule = True
    for i, move in enumerate(moves):
        if not is_valid_move(move):
            return 6, board  # Invalid move

        x, y = convert_move_to_coords(move)

        if i == 1 and _allow_swap_rule:
            if move == moves[0]:  # If the second move matches the first
                board[y, x] = 0  # Removing red's move
                y, x = x, y  # Swapping coordinates
                swap_move = coords_to_move(x, y)
                board[y, x] = 2  # Placing blue stone
                applied_moves = [(x, y)]
            else:
                board[y, x] = 2  # Normal blue move
                applied_moves.append((x, y))
        else:
            if board[y, x] != 0:
                return 5, board  # Occupied cell
            if i >= 2 and (x, y) in applied_moves:
                return 5, board  # Occupied cell
            board[y, x] = 1 if i % 2 == 0 else 2
            applied_moves.append((x, y))

    return None, board

def parse_moves_sequence(sequence):
    # Parsing a move sequence from a string format
    return sequence.split()

def get_neighbors(i, j, size):
    # Getting all neighboring cells within the board boundaries
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    return [(i + di, j + dj) for di, dj in directions if 0 <= i + di < size and 0 <= j + dj < size]

def dfs(board, start, player, goal_edge):
    # Performing depth-first search to check if a player connects sides
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
    # Checking if red has a path from top to bottom
    size = len(board)
    for j in range(size):
        if board[0, j] == 1:  # Checking the top edge
            if dfs(board, (0, j), 1, lambda i, j: i == size - 1):
                return True
    return False

def check_winner_blue(board):  # Blue (2) connects left and right
    # Checking if blue has a path from left to right
    size = len(board)
    for i in range(size):
        if board[i, 0] == 2:  # Checking the left edge
            if dfs(board, (i, 0), 2, lambda i, j: j == size - 1):
                return True
    return False

def check_winner(board):
    # Checking if there is a winner on the board
    if check_winner_red(board):
        return 3
    elif check_winner_blue(board):
        return 4
    else:
        return 2

def play_game(moves_sequence):
    # Playing a game with a given sequence of moves and determining the winner
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
tokenizer_path = os.path.join(base_path, "tokenizer")
logs_dir = os.path.join(base_path, "logs")
all_games_df = pd.DataFrame(columns=['Red Checkpoint', 'Blue Checkpoint', 'Game Moves', 'Winner'])

# Function to filter and sort checkpoints
def filter_and_sort_checkpoints(checkpoints):
    # Filtering checkpoints to keep only those containing 'checkpoint-'
    filtered = [cp for cp in checkpoints if 'checkpoint-' in cp]
    
    # Sorting by the number after 'checkpoint-'
    sorted_checkpoints = sorted(filtered, key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    
    return sorted_checkpoints

# List of checkpoint directories
red_checkpoints = [os.path.join(red_checkpoints_dir, cp) for cp in os.listdir(red_checkpoints_dir)]
blue_checkpoints = [os.path.join(blue_checkpoints_dir, cp) for cp in os.listdir(blue_checkpoints_dir)]

# Applying the function
sorted_red_checkpoints = filter_and_sort_checkpoints(red_checkpoints)
sorted_blue_checkpoints = filter_and_sort_checkpoints(blue_checkpoints)
sum_checkpoints = sorted_red_checkpoints + sorted_blue_checkpoints

# Total number of games
total_games = len(red_checkpoints) * len(blue_checkpoints) * 121

# Loading tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
special_tokens = {'pad_token': '<pad>', 'unk_token': '<unk>'}
tokenizer.add_special_tokens(special_tokens)

# Defining all possible positions with whitespace
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

def predict_next_tokens(model, moves_sequence, top_k=125):
    # Predicting the next tokens for a given move sequence
    input_text = moves_sequence
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
    top_k_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_k_indices[0].tolist()]
    top_k_probs = top_k_probs[0].tolist()
    return list(top_k_tokens)

# Creating a 3D cube structure for games and stats
def shape_the_cube():
    games_cube = [[[None for _ in range(len(sorted_blue_checkpoints) * 121 + 1)] 
                    for _ in range(3)] 
                    for _ in range(len(sorted_red_checkpoints) + len(sorted_blue_checkpoints))]
    
    stats_cube = [[[0, 0] for _ in range(len(sorted_blue_checkpoints) + 1)]
                    for _ in range(len(sorted_red_checkpoints) + 1)]
    
    stats_cube[0][0] = ["red_vertically", "blue_horizontally"]

    number_to_checkpoint = []
    # Adding values for each blue player
    for blue_player in range(len(sorted_blue_checkpoints)):
        # Adding red checkpoints for each blue player
        games_cube[blue_player][0][0] = blue_player
        games_cube[blue_player][1][0] = 2  # Move number
        games_cube[blue_player][2][0] = len(sorted_red_checkpoints) * 121  # Number of games in the list
        number_to_checkpoint.append(sorted_blue_checkpoints[blue_player])
        stats_cube[0][blue_player + 1] = sorted_blue_checkpoints[blue_player]
        # Adding 121 moves for each red checkpoint
        for red_checkpoint in range(len(sorted_red_checkpoints)):
            for _ in range(len(positions)):
                games_cube[blue_player][0][1 + red_checkpoint * 121 + _] = red_checkpoint + len(sorted_blue_checkpoints)
                games_cube[blue_player][1][1 + red_checkpoint * 121 + _] = positions[_]
    for red_player in range(len(sorted_red_checkpoints)):
        games_cube[red_player+len(sorted_blue_checkpoints)][0][0] = red_player + len(sorted_blue_checkpoints)
        games_cube[red_player+len(sorted_blue_checkpoints)][1][0] = 3  # Move number
        games_cube[red_player+len(sorted_blue_checkpoints)][2][0] = 0  # Number of games in the list
        number_to_checkpoint.append(sorted_red_checkpoints[red_player])
        stats_cube[red_player + 1][0] = sorted_red_checkpoints[red_player]
    return games_cube, number_to_checkpoint, stats_cube

games_cube, number_to_checkpoint, stats_cube = shape_the_cube()

def count_remaining_games():
    # Calculating the number of remaining games
    sum = 0 
    for i in range(len(games_cube)):
        sum += games_cube[i][2][0]
    return sum

def update_checkpoint_stats(map, games_map, main_checkpoint, opponent_checkpoint, game_to_check, result):
    # Updating checkpoint statistics and game results map
    red_checkpoint, blue_checkpoint = (main_checkpoint, opponent_checkpoint) if main_checkpoint > opponent_checkpoint else (opponent_checkpoint, main_checkpoint)
    if result == 3:
        map[red_checkpoint - len(sorted_blue_checkpoints) + 1][blue_checkpoint + 1][0] += 1
        games_map = pd.concat([all_games_df, pd.DataFrame([{
                        'Red Checkpoint': red_checkpoint,
                        'Blue Checkpoint': blue_checkpoint,
                        'Game Moves': game_to_check,
                        'Winner': 'Red'
                    }])], ignore_index=True)
    else:
        map[red_checkpoint - len(sorted_blue_checkpoints)+ 1][blue_checkpoint + 1][1] += 1
        games_map = pd.concat([all_games_df, pd.DataFrame([{
                        'Red Checkpoint': red_checkpoint,
                        'Blue Checkpoint': blue_checkpoint,
                        'Game Moves': game_to_check,
                        'Winner': 'Blue'
                    }])], ignore_index=True)
    return map, games_map

def preload_models():
    models = []
    # Loading models
    for i in range(len(games_cube)):
        models.append(load_model(number_to_checkpoint[i]))
    return models

def main():
    global all_games_df, stats_cube
    move_number = 1
    games_left = 1
    start_time = time.time()
    models = preload_models()
    pbar = tqdm(total=total_games, desc="Processing Games", unit="game")
    while games_left != 0:
        move_number += 2
        games_left = count_remaining_games()
        for checkpoint_to_play in range(len(games_cube)):
            model = models[checkpoint_to_play]
            for game_number in range(games_cube[checkpoint_to_play][2][0]):
                game_number += 1
                games_cube[checkpoint_to_play][2][game_number] = predict_next_tokens(model, games_cube[checkpoint_to_play][1][game_number])
            for game_number in range(games_cube[checkpoint_to_play][2][0]):
                game_number += 1
                game_result = 0
                move_index = 0
                while game_result < 2 or game_result > 4:
                    game_to_check = str(games_cube[checkpoint_to_play][1][game_number]) + ' ' + str(games_cube[checkpoint_to_play][2][game_number][move_index])
                    game_result = play_game(game_to_check)
                    move_index += 1
                games_cube[checkpoint_to_play][1][game_number] = game_to_check
                if game_result == 2:
                    games_cube[games_cube[checkpoint_to_play][0][game_number]][2][0] += 1
                    games_cube[games_cube[checkpoint_to_play][0][game_number]][1][games_cube[games_cube[checkpoint_to_play][0][game_number]][2][0]] = game_to_check
                    games_cube[games_cube[checkpoint_to_play][0][game_number]][0][games_cube[games_cube[checkpoint_to_play][0][game_number]][2][0]] = games_cube[checkpoint_to_play][0][0]
                else:  # Red wins
                    stats_cube, all_games_df = update_checkpoint_stats(stats_cube, all_games_df, checkpoint_to_play, games_cube[checkpoint_to_play][0][game_number], game_to_check, game_result)
                # Updating progress bar after processing each game
                pbar.update(total_games - games_left - pbar.n)  
                pbar.set_postfix(move=move_number)
            games_cube[checkpoint_to_play][2][0] = 0    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"exec time: {execution_time} sec")
    # Closing progress bar after processing
    pbar.close()
    all_games_df.to_csv(os.path.join(logs_dir, 'all_games_results.csv'), index=False, sep=';')
    with open(os.path.join(logs_dir, 'games_matrix.csv'), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(stats_cube)

if __name__ == "__main__":
    main()
