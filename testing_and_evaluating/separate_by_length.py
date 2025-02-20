import os

base_dir = os.path.join(os.path.dirname(__file__), '..')
input_file = os.path.join(base_dir, 'red_wins_red_training_set.txt')

output_file1 = os.path.join(base_dir, 'red_upto50.txt')
output_file2 = os.path.join(base_dir, 'red_5080.txt')
output_file3 = os.path.join(base_dir, 'red_over80.txt')

# Opening output files and process input file line by line
with open(output_file1, 'w', encoding='utf-8') as f1, \
     open(output_file2, 'w', encoding='utf-8') as f2, \
     open(output_file3, 'w', encoding='utf-8') as f3:

    # Reading input file line by line and counting spaces
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            num_spaces = line.count(' ')
            if num_spaces <= 50:
                f1.write(line)
            elif 50 < num_spaces <= 80:
                f2.write(line)
            else:
                f3.write(line)
