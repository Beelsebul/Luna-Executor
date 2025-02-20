import os
import pandas as pd


base_dir = os.path.join(os.path.dirname(__file__), '..')
input_file = os.path.join(base_dir, 'logs', 'red_training_set.csv')
output_blue = os.path.join(base_dir, 'blue_wins_red_training_set.txt')
output_red = os.path.join(base_dir, 'red_wins_red_training_set.txt')

df = pd.read_csv(input_file, header=None)

blue_games = df[df[1].str.contains("Blue", case=False, na=False)]
red_games = df[df[1].str.contains("Red", case=False, na=False)]
blue_games[0].to_csv(output_blue, index=False, header=False)
red_games[0].to_csv(output_red, index=False, header=False)

print(f"Saved as: {output_blue} and {output_red}")
