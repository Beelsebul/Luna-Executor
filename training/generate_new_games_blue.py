import numpy as np
import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm 
import pandas as pd
from pandas import concat
from collections import deque
import time
from functools import lru_cache

def create_board():
    # Creating an empty game board as an 11x11 grid
    return np.zeros((11, 11), dtype=int)

def convert_move_to_coords(move):
    # Converting a move in notation (e.g., "A1") to board coordinates (x, y)
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

def modify_and_multiply(list):
    # Padding the list with ones if it has fewer than 121 elements
    if len(list) <= 121:
        list += [1] * (122 - len(list))
    
    # Calculating the product of all elements
    product = np.prod(list)
    
    return list, product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_path = os.path.join(os.path.dirname(__file__), '..')
red_checkpoint_path = os.path.join(base_path, 'mini_red_test', 'checkpoint-765960_ft_r_gen_4')
blue_checkpoint_path = os.path.join(base_path, 'mini_blue_test', 'checkpoint-589095_ft_gen_3_b')
tokenizer_path = os.path.join(base_path, 'tokenizer')
logs_dir = os.path.join(base_path, 'logs')

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = '<|pad|>'
tokenizer.bos_token = '<|startoftext|>'
tokenizer.eos_token = '<|endoftext|>'
tokenizer.unk_token = '<|unk|>'

def load_model(checkpoint_path):
    # Loading a model from a checkpoint
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model

def predict_next_tokens(model, moves_sequences, top_k=148):
    # Predicting the next token options for a list of move sequences
    input_texts = ['<|startoftext|> ' + seq.strip() for seq in moves_sequences]
    inputs = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
    
    top_k_tokens = []
    for i in range(len(moves_sequences)):
        tokens = [tokenizer.decode([token_id]).strip() for token_id in top_k_indices[i].tolist()]
        top_k_tokens.append(tokens)
    
    return top_k_tokens

def update_checkpoint_stats(games_map, game_to_check, result):
    # Updating the game statistics in the checkpoint with the latest game and result
    if result == 3:
        games_map = pd.concat([games_map, pd.DataFrame([{
                        'Game Moves': game_to_check,
                        'Winner': 'Red'
                    }])], ignore_index=True)
    else:
        games_map = pd.concat([games_map, pd.DataFrame([{
                        'Game Moves': game_to_check,
                        'Winner': 'Blue'
                    }])], ignore_index=True)
    return games_map

def main():
    # Setting up the main loop to simulate games and gather results
    blue_queue = deque()
    red_queue = deque([''])
    all_games_df = pd.DataFrame(columns=['Game Moves', 'Winner'])
    red_checkpoint = load_model(red_checkpoint_path)
    blue_checkpoint = load_model(blue_checkpoint_path)
    mul_moves = [8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
  # mul_moves = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    # Extending moves and calculating their product
    mul_moves_ext, product = modify_and_multiply(mul_moves)
    finished_games = 0
    move = 0
    start_time = time.time()
    pbar = tqdm(total=product, desc="Processing Games", unit="game")
    fixed_batch_size = 500

    while finished_games < product:
        # Processing red moves to avoid bugs
        if move > 124:
            print("Loop interrupted: move exceeds.")
            break
        
        while len(red_queue) > 0:
            batch_size = min(fixed_batch_size, len(red_queue))
            games_to_generate = [red_queue.popleft() for _ in range(batch_size)]
            all_possible_moves = predict_next_tokens(red_checkpoint, games_to_generate)

            for i, game_to_generate in enumerate(games_to_generate):
                possible_moves = all_possible_moves[i]
                game_result = 0
                move_index = 0
                for n in range(mul_moves_ext[move]):
                    while game_result < 2 or game_result > 4:
                        game_to_check = str(game_to_generate) + ' ' + str(possible_moves[move_index])
                        game_result = play_game(game_to_check)
                        move_index += 1
                    if game_result == 2:
                        blue_queue.append(game_to_check)
                    else:
                        pbar.update(1)
                        pbar.set_postfix(move=move)
                        all_games_df = update_checkpoint_stats(all_games_df, game_to_check, game_result)
                        finished_games += 1
                    game_result = 0

        move += 1
        pbar.update(0)
        pbar.set_postfix(move=move)
        # Interrupting loop if move exceeds 121
        if move > 124:
            print("Loop interrupted: move exceeds 121.")
            break

        # Processing blue moves
        while len(blue_queue) > 0:
            batch_size = min(fixed_batch_size, len(blue_queue))
            games_to_generate = [blue_queue.popleft() for _ in range(batch_size)]
            all_possible_moves = predict_next_tokens(blue_checkpoint, games_to_generate)

            for i, game_to_generate in enumerate(games_to_generate):
                possible_moves = all_possible_moves[i]
                game_result = 0
                move_index = 0
                for n in range(mul_moves_ext[move]):
                    while game_result < 2 or game_result > 4:
                        game_to_check = str(game_to_generate) + ' ' + str(possible_moves[move_index])
                        game_result = play_game(game_to_check)
                        move_index += 1
                    if game_result == 2:
                        red_queue.append(game_to_check)
                    else:
                        pbar.update(1)
                        pbar.set_postfix(move=move)
                        all_games_df = update_checkpoint_stats(all_games_df, game_to_check, game_result)
                        finished_games += 1
                    game_result = 0

        move += 1
        pbar.update(0)
        pbar.set_postfix(move=move)
        # Interrupting loop if move exceeds 121
        if move > 124:
            print("Loop interrupted: move exceeds 121.")
            break

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n Execution time: {execution_time} sec")
    all_games_df.to_csv(os.path.join(logs_dir, 'blue_training_set.csv'), index=False, sep=',')

if __name__ == "__main__":
    main()
