This section provides a description of all elements in this project, including code, graphs, and models.

Folders available for review:

Graphs - Contains visual representations of model testing as well as some loss graphs from training.
Training - Contains all scripts related to model creation and game generation using already trained models.
testing_and_evaluating - Contains all scripts for data processing, evaluation, and model testing.
Website - Contains files for initializing a local server as well as a page template for playing the board game.
minihex_red and minihex_blue - Contain checkpoints of different generations.



Files in the Training folder:

tokenizer.py - Used for training and creating a tokenizer, saving it in the tokenizer folder.

train_minihex_red.py - Responsible for training an initial model based on starting games won by the red player, with random initial weight initialization. Saves the model to the minihex_red folder and training data in logs.

train_minihex_blue.py - Similar, but for the blue player. Saves the model to the minihex_blue folder and training data in logs.

generate_new_games_red.py - Responsible for generating new games between two models by selecting more than one most likely move for the red model. Generated games are saved to a CSV file in logs.

generate_new_games_blue.py - Similar, but for the blue player. Generated games are saved to a CSV file in logs.

fine_tune_minihex_red.py - Responsible for fine-tuning the model based on new games won by the red player, loading weights from the previous model. Saves the model to the minihex_red folder and training data in logs.

fine_tune_minihex_blue.py - Similar, but for the blue player. Saves the model to the minihex_blue folder and training data in logs.




Files in the Testing_and_Evaluating folder:

shorten_or_shuffle.py - Shuffles data in a text file or randomly selects a specified number of words. Saves the file to the hex_agony folder.

separate_red_and_blue.py - Separates games from a CSV file into two files with red and blue winners. Saves files in the hex_agony folder.

separate_by_length.py - Divides games in a TXT file into three files based on the number of moves in the game. Used to monitor the length of generated games across generations. Saves files in the hex_agony folder.

test_model.py - The first model testing implementation, which provides the top 5 predictions for a game entered in the input field.

test_tokenizer.py - Created to view and verify the tokenization process.

testwinner.py - Tests models by playing them against each other with each of the 121 possible moves and creates a CSV matrix of game results. Saves the CSV file with the results matrix as well as a CSV file with the played games.

testwinner_gpt_optimized.py - Similar to testwinner.py, but optimized by ChatGPT and works several times faster.

heatmap.py - Displays the same results matrix but represents relationships with colors for easier data visualization.

elo_compute.py - Calculates model rankings using the Elo rating based on the matrix obtained from testwinner, simplifying model comparison from matrix form to numeric representation.




Files in the Graphs folder:

data_for_graphs/loss folder contains CSV files with mean_loss, validation_loss, and training_loss metrics from fine-tuning on initial data (loss is meaningful only for fine-tuning on this type of data).

poorly_trained_checkpoints.png - 0,1 - first generation, 2 and 3 - second generation (unsuccessful), 4 and 5 - third generation (unsuccessful). The comparison graph shows how one third-generation model can simultaneously win against the second generation and lose against the first generation, which the second generation defeats. Unsuccessful generations also show an increase in average game length, while successful generations show a decrease, although these data are not displayed on the heatmap.

heatmap-final.png - The final table compares the top three models of each generation against each other.




Files in the Website folder:
app.py - to start the webserver which then enables user to test the model via visual interface.
static and templates contain files required to build visual interface including the game logics and board Generation (the board is resizable)


