import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.join(os.path.dirname(__file__), '..')
file_path = os.path.join(base_dir, 'logs', 'games_matrix.csv')

with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = f.readlines()

# Initializing lists to store results
results = []

# Regular expression to extract numbers from the result lines
pattern = re.compile(r'\[(\d+),\s*(\d+)\]')

# Processing each line in raw_data and extracting the victory numbers
for line in raw_data[1:]:
    matches = pattern.findall(line)
    if matches:
        # Converting the string results to integer pairs and storing them
        results.append([list(map(int, match)) for match in matches])

# Converting the list to a NumPy array for easier manipulation
results_array = np.array(results)

# Separating red and blue victory results
red_wins = results_array[:, :, 0]
blue_wins = results_array[:, :, 1]

# Creating the heatmap for the red and blue wins
fig, ax = plt.subplots(figsize=(10, 8))

# Generating a heatmap for red wins and blue wins combined with annotations
heatmap_data = red_wins / (red_wins + blue_wins)  # Proportion of red victories over total matches

sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="seismic", center=0.5, ax=ax)

# Adding labels and title
ax.set_title("Proportion of Red Wins vs Blue Wins")
ax.set_xlabel("Blue Models")
ax.set_ylabel("Red Models")

plt.tight_layout()
plt.show()
