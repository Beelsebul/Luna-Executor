import os
import pandas as pd
import numpy as np

# Loading the CSV file to process Elo ratings
base_dir = os.path.join(os.path.dirname(__file__), '..')
file_path = os.path.join(base_dir, 'logs', 'games_matrix.csv')
games_matrix = pd.read_csv(file_path, delimiter=';', skipinitialspace=True, on_bad_lines='skip')

# Extracting the list of checkpoints (models) and results data
# The first column contains red checkpoints, subsequent columns contain the results
red_checkpoints = games_matrix.iloc[:, 0].unique()
blue_checkpoints = games_matrix.columns[1:]

# Initializing Elo ratings for all checkpoints, starting with a base rating of 1000
elo_ratings_red = {checkpoint: 1000 for checkpoint in red_checkpoints}
elo_ratings_blue = {checkpoint: 1000 for checkpoint in blue_checkpoints}

# Function to calculate the expected score between two Elo ratings
# This calculates the probability that player A (with Elo elo_a) will win against player B (elo_b)
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# Function to update Elo ratings based on the outcome
# K-factor determines the sensitivity of rating changes; higher K values mean larger adjustments per game
def update_elo(elo_a, elo_b, score_a, k=32):
    expected_a = expected_score(elo_a, elo_b)
    # Elo formula: new_rating = current_rating + K * (actual_score - expected_score)
    new_elo_a = elo_a + k * (score_a - expected_a)
    return new_elo_a

# Processing each row in the dataset to update Elo ratings for each pair of red and blue checkpoints
for index, row in games_matrix.iterrows():
    red_checkpoint = row[0]
    for blue_checkpoint, result in row[1:].items():
        if pd.notna(result):
            # Extracting wins from result string in the format "[red_wins, blue_wins]"
            try:
                red_wins, blue_wins = map(int, result.strip('[]').split(','))
            except ValueError:
                continue
            
            total_games = red_wins + blue_wins
            if total_games == 0:
                continue  # Skip if no games were played
            
            # Calculating scores for Elo update as proportions of wins for each player
            score_red = red_wins / total_games  # Red's proportion of total wins
            score_blue = blue_wins / total_games  # Blue's proportion of total wins

            # Updating Elo ratings for red and blue checkpoints
            elo_ratings_red[red_checkpoint] = update_elo(elo_ratings_red[red_checkpoint], elo_ratings_blue[blue_checkpoint], score_red)
            elo_ratings_blue[blue_checkpoint] = update_elo(elo_ratings_blue[blue_checkpoint], elo_ratings_red[red_checkpoint], score_blue)

# Sorting the Elo ratings in descending order to find the top-rated checkpoints
sorted_elo_ratings_red = sorted(elo_ratings_red.items(), key=lambda x: x[1], reverse=True)
sorted_elo_ratings_blue = sorted(elo_ratings_blue.items(), key=lambda x: x[1], reverse=True)

# Displaying the top Elo ratings for red and blue checkpoints
print("Top Red Checkpoints by Elo Rating:")
for checkpoint, elo in sorted_elo_ratings_red:
    print(f"{checkpoint}: Elo {elo:.2f}")

print("\nTop Blue Checkpoints by Elo Rating:")
for checkpoint, elo in sorted_elo_ratings_blue:
    print(f"{checkpoint}: Elo {elo:.2f}")  
