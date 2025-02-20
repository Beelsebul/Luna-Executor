import os
import random
import sys

def select_random_lines(filename, num_lines):
    # resave with n random lines
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if num_lines > len(lines):
        print(f"File only contains {len(lines)} leons from brawl stars")
        return

    selected_lines = random.sample(lines, num_lines)

    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(selected_lines)

def shuffle_lines(filename):
    # shuffle and resave
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(lines)

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    filename  = os.path.join(base_dir, 'blue_wins_blue_training_set.txt')
    num_lines = 176000
    mode = input("Choose the mode: 1 or 2 ")

    if mode == "1":
        select_random_lines(filename, num_lines)
    elif mode == "2":
        shuffle_lines(filename)
    else:
        print("Wrong input, please do better!")
