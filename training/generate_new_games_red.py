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
    # Defining move multiplier list for game generation
    mul_moves = [8, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
  # mul_moves = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    mul_moves_ext, product = modify_and_multiply(mul_moves)
    finished_games = 0
    move = 0
    start_time = time.time()
    pbar = tqdm(total=product, desc="Processing Games", unit="game")
    # Set a fixed batch size
    fixed_batch_size = 500

    while finished_games < product:
        # Processing red moves
        if move > 124:
            print("Loop interrupted: move exceeded.")
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
        # Interrupting the loop if move exceeds 121
        if move > 124:
            print("Loop interrupted: move exceeded.")
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
        # Interrupting the loop if move exceeds 121
        if move > 124:
            print("Loop interrupted: move exceeded.")
            break


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n Execution time: {execution_time} sec")
    all_games_df.to_csv(os.path.join(logs_dir, 'red_training_set.csv'), index=False, sep=',')

if __name__ == "__main__":
    main()



































# import numpy as np 
# import os
# import torch
# from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
# from tqdm import tqdm 
# import pandas as pd
# from collections import deque
# import time

# # 1. Определяем устройство глобально
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def create_board(size=11):
#     """Создает пустую игровую доску."""
#     return np.zeros((size, size), dtype=int)

# def convert_move_to_coords(move):
#     """Преобразует ход из формата 'A1' в координаты (x, y)."""
#     letter = move[0].upper()
#     number = int(move[1:]) - 1
#     x = ord(letter) - ord('A')
#     y = number
#     return x, y

# def coords_to_move(x, y):
#     """Преобразует координаты (x, y) обратно в формат хода 'A1'."""
#     letter = chr(x + ord('A'))
#     number = y + 1
#     return f"{letter}{number}"

# def is_valid_move(move, size=11):
#     """Проверяет, является ли ход допустимым."""
#     if len(move) < 2:
#         return False
#     letter = move[0].upper()
#     if not 'A' <= letter <= chr(ord('A') + size - 1):
#         return False
#     try:
#         number = int(move[1:])
#         return 1 <= number <= size
#     except ValueError:
#         return False

# def apply_move(board, move, player):
#     """Применяет ход к доске."""
#     x, y = convert_move_to_coords(move)
#     if board[y, x] != 0:
#         return False  # Клетка занята
#     board[y, x] = player
#     return True

# def get_neighbors(i, j, size):
#     """Возвращает соседние клетки для заданной клетки (i, j)."""
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
#     return [(i + di, j + dj) for di, dj in directions if 0 <= i + di < size and 0 <= j + dj < size]

# def dfs(board, start, player, goal_edge):
#     """Выполняет поиск в глубину для проверки соединения."""
#     size = len(board)
#     stack = [start]
#     visited = set()

#     while stack:
#         current = stack.pop()
#         if current not in visited:
#             visited.add(current)
#             i, j = current

#             if goal_edge(i, j):
#                 return True

#             for ni, nj in get_neighbors(i, j, size):
#                 if board[ni, nj] == player and (ni, nj) not in visited:
#                     stack.append((ni, nj))

#     return False

# def check_winner_red(board):
#     """Проверяет победу Красного (соединение верх и низ)."""
#     size = len(board)
#     for j in range(size):
#         if board[0, j] == 1:  # Проверяем верхний край
#             if dfs(board, (0, j), 1, lambda i, j: i == size - 1):
#                 return True
#     return False

# def check_winner_blue(board):
#     """Проверяет победу Синего (соединение левого и правого края)."""
#     size = len(board)
#     for i in range(size):
#         if board[i, 0] == 2:  # Проверяем левый край
#             if dfs(board, (i, 0), 2, lambda i, j: j == size - 1):
#                 return True
#     return False

# def check_winner(board):
#     """Определяет победителя или продолжение игры."""
#     if check_winner_red(board):
#         return 3  # Победа Красного
#     elif check_winner_blue(board):
#         return 4  # Победа Синего
#     else:
#         return 2  # Игра продолжается

# def modify_and_multiply(lst):
#     """
#     Дополняет список единицами до длины 121 и вычисляет произведение всех элементов.
    
#     Args:
#         lst (list): Исходный список.
    
#     Returns:
#         tuple: (дополненный список, произведение элементов)
#     """
#     if len(lst) < 121:
#         lst += [1] * (121 - len(lst))
#     elif len(lst) > 121:
#         lst = lst[:121]
    
#     product = np.prod(lst)
    
#     return lst, product

# def predict_next_tokens(model, moves_sequences, top_k=148):
#     """
#     Предсказывает следующие ходы для заданных последовательностей ходов.
    
#     Args:
#         model (GPT2LMHeadModel): Загруженная модель GPT-2.
#         moves_sequences (list): Список последовательностей ходов.
#         top_k (int): Количество топовых ходов для предсказания.
    
#     Returns:
#         list: Список списков возможных ходов для каждой последовательности.
#     """
#     input_texts = ['<|startoftext|> ' + seq.strip() for seq in moves_sequences]
#     inputs = tokenizer.batch_encode_plus(
#         input_texts, 
#         return_tensors='pt', 
#         padding=True, 
#         truncation=True,
#         max_length=512  # Устанавливаем max_length для избежания предупреждений
#     )
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)

#     next_token_logits = outputs.logits[:, -1, :]
#     next_token_probs = torch.softmax(next_token_logits, dim=-1)
#     top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)

#     top_k_tokens = []
#     for i in range(len(moves_sequences)):
#         tokens = [tokenizer.decode([token_id]).strip() for token_id in top_k_indices[i].tolist()]
#         top_k_tokens.append(tokens)

#     return top_k_tokens

# def update_checkpoint_stats(all_games_results, game_to_check, result):
#     """
#     Обновляет список результатов игр.
    
#     Args:
#         all_games_results (list): Список результатов игр.
#         game_to_check (str): Последовательность ходов игры.
#         result (int): Результат игры.
    
#     Returns:
#         list: Обновленный список результатов игр.
#     """
#     if result == 3:
#         all_games_results.append({
#             'Game Moves': game_to_check,
#             'Winner': 'Red'
#         })
#     elif result == 4:
#         all_games_results.append({
#             'Game Moves': game_to_check,
#             'Winner': 'Blue'
#         })
#     return all_games_results

# def load_model(checkpoint_path):
#     """
#     Загружает модель GPT-2 из указанного пути.
    
#     Args:
#         checkpoint_path (str): Путь к чекпойнту модели.
    
#     Returns:
#         GPT2LMHeadModel: Загруженная модель.
#     """
#     model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
#     model.resize_token_embeddings(len(tokenizer))
#     model.to(device)
#     model.eval()
#     return model

# def main():
#     # Инициализация путей
#     base_path = r"C:\Users\Timur\Downloads\hex_agony"
#     red_checkpoint_path = os.path.join(base_path, "mini_red_test", "checkpoint-638300")
#     blue_checkpoint_path = os.path.join(base_path, "mini_blue_test", "checkpoint-654550")
#     tokenizer_path = os.path.join(base_path, "tokenizer")
#     logs_dir = os.path.join(base_path, "logs")
    
#     # Инициализация токенизатора
#     global tokenizer  # Объявляем глобально для использования в predict_next_tokens
#     tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
#     tokenizer.pad_token = '<|pad|>'
#     tokenizer.bos_token = '<|startoftext|>'
#     tokenizer.eos_token = '<|endoftext|>'
#     tokenizer.unk_token = '<|unk|>'

#     # Загрузка моделей
#     red_model = load_model(red_checkpoint_path)
#     blue_model = load_model(blue_checkpoint_path)

#     # Определение mul_moves и вычисление product
#     mul_moves = [20, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
#     mul_moves_ext, product = modify_and_multiply(mul_moves)
#     print(f"Total games to generate: {product}")

#     # Инициализация очередей
#     red_queue = deque()
#     blue_queue = deque()

#     # Инициализация первой игры (пустая последовательность ходов, пустая доска, move_number=0)
#     initial_game = {'moves': '', 'board': create_board(), 'move_number': 0, 'swapped': False}
#     red_queue.append(initial_game)

#     # Инициализация списка для результатов игр
#     all_games_results = []

#     # Инициализация счетчиков
#     finished_games = 0
#     start_time = time.time()

#     # Инициализация прогресс-бара
#     pbar = tqdm(total=product, desc="Processing Games", unit="game")

#     # Определение фиксированного размера батча
#     fixed_batch_size = 1000  # Можно настроить в зависимости от доступной памяти

#     while finished_games < product and (red_queue or blue_queue):
#         # Обработка ходов Красного
#         if red_queue:
#             batch_size = min(fixed_batch_size, len(red_queue))
#             games_batch = [red_queue.popleft() for _ in range(batch_size)]
#             move_sequences = [game['moves'] for game in games_batch]
#             top_k_moves = predict_next_tokens(red_model, move_sequences)

#             for idx, game in enumerate(games_batch):
#                 possible_moves = top_k_moves[idx]
#                 current_move_number = game['move_number']
#                 max_branches = mul_moves_ext[current_move_number]

#                 branches_applied = 0
#                 move_idx = 0

#                 while branches_applied < max_branches and move_idx < len(possible_moves):
#                     move = possible_moves[move_idx]
#                     move_idx += 1

#                     if not is_valid_move(move):
#                         continue

#                     # Handle swap rule on the second move
#                     if current_move_number == 1 and not game.get('swapped', False):
#                         first_move = game['moves'].split()[0] if game['moves'] else None
#                         if first_move and move == first_move:
#                             # Apply swap
#                             new_board = game['board'].copy()
#                             first_x, first_y = convert_move_to_coords(first_move)
#                             new_board[first_y, first_x] = 0  # Remove first Red move
#                             swap_x, swap_y = first_x, first_y
#                             new_board[swap_y, swap_x] = 2  # Set Blue's move
#                             new_moves_sequence = coords_to_move(swap_x, swap_y)
#                             new_game = {
#                                 'moves': new_moves_sequence,
#                                 'board': new_board,
#                                 'move_number': 1,
#                                 'swapped': True
#                             }
#                             # Check for winner
#                             winner = check_winner(new_board)
#                             if winner == 3:
#                                 # Red wins (unlikely in swap)
#                                 all_games_results.append({'Game Moves': new_moves_sequence, 'Winner': 'Red'})
#                                 pbar.update(1)
#                                 finished_games +=1
#                             elif winner == 4:
#                                 # Blue wins
#                                 all_games_results.append({'Game Moves': new_moves_sequence, 'Winner': 'Blue'})
#                                 pbar.update(1)
#                                 finished_games +=1
#                             else:
#                                 # Game continues, add to blue queue
#                                 blue_queue.append(new_game)
#                             branches_applied +=1
#                             continue

#                     # Apply move
#                     new_board = game['board'].copy()
#                     success = apply_move(new_board, move, 1)  # Player 1: Red
#                     if not success:
#                         continue  # Invalid move, skip

#                     # Update move sequence
#                     new_moves_sequence = (game['moves'] + ' ' + move).strip()

#                     # Check for winner
#                     winner = check_winner(new_board)

#                     if winner == 3:
#                         # Red wins
#                         all_games_results.append({'Game Moves': new_moves_sequence, 'Winner': 'Red'})
#                         pbar.update(1)
#                         finished_games +=1
#                     elif winner == 4:
#                         # Blue wins (unlikely here)
#                         all_games_results.append({'Game Moves': new_moves_sequence, 'Winner': 'Blue'})
#                         pbar.update(1)
#                         finished_games +=1
#                     else:
#                         # Game continues, add to blue queue
#                         new_game = {
#                             'moves': new_moves_sequence,
#                             'board': new_board,
#                             'move_number': current_move_number + 1,
#                             'swapped': game.get('swapped', False)
#                         }
#                         blue_queue.append(new_game)
#                     branches_applied +=1

#         # Обработка ходов Синего
#         if blue_queue:
#             batch_size = min(fixed_batch_size, len(blue_queue))
#             games_batch = [blue_queue.popleft() for _ in range(batch_size)]
#             move_sequences = [game['moves'] for game in games_batch]
#             top_k_moves = predict_next_tokens(blue_model, move_sequences)

#             for idx, game in enumerate(games_batch):
#                 possible_moves = top_k_moves[idx]
#                 current_move_number = game['move_number']
#                 max_branches = mul_moves_ext[current_move_number]

#                 branches_applied = 0
#                 move_idx = 0

#                 while branches_applied < max_branches and move_idx < len(possible_moves):
#                     move = possible_moves[move_idx]
#                     move_idx +=1

#                     if not is_valid_move(move):
#                         continue

#                     # Apply move
#                     new_board = game['board'].copy()
#                     success = apply_move(new_board, move, 2)  # Player 2: Blue
#                     if not success:
#                         continue  # Invalid move, skip

#                     # Update move sequence
#                     new_moves_sequence = (game['moves'] + ' ' + move).strip()

#                     # Check for winner
#                     winner = check_winner(new_board)

#                     if winner == 4:
#                         # Blue wins
#                         all_games_results.append({'Game Moves': new_moves_sequence, 'Winner': 'Blue'})
#                         pbar.update(1)
#                         finished_games +=1
#                     elif winner == 3:
#                         # Red wins (unlikely here)
#                         all_games_results.append({'Game Moves': new_moves_sequence, 'Winner': 'Red'})
#                         pbar.update(1)
#                         finished_games +=1
#                     else:
#                         # Game continues, add to red queue
#                         new_game = {
#                             'moves': new_moves_sequence,
#                             'board': new_board,
#                             'move_number': current_move_number +1,
#                             'swapped': game.get('swapped', False)
#                         }
#                         red_queue.append(new_game)
#                     branches_applied +=1

#         # Ранний выход, если обе очереди пусты
#         if not red_queue and not blue_queue:
#             break

#     # Закрываем прогресс-бар
#     pbar.close()

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"\nExecution time: {execution_time:.2f} sec")

#     # Сохранение результатов в CSV
#     all_games_df = pd.DataFrame(all_games_results)
#     os.makedirs(logs_dir, exist_ok=True)
#     all_games_df.to_csv(os.path.join(logs_dir, 'fall_games_set.csv'), index=False, sep=',')

# if __name__ == "__main__":
#     main()
