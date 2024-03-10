import os
import json

# Path to the folder containing JSON files
folder_path = '/Users/ayushsahni/Desktop/Documents/Projects/hackmerced/transcribed'

# Initialize an empty list to store input-output pairs
input_output_pairs = []

# Iterate over each JSON file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        # Load JSON data
        with open(file_path, 'r') as f:
            conversations = json.load(f)
        # Preprocess conversations
        for conv in conversations:
            dialogue = conv['dialogue']
            for i in range(len(dialogue) - 1):
                input_text = dialogue[i]
                target_text = dialogue[i + 1]
                input_output_pairs.append((input_text, target_text))

# Save preprocessed data
with open('preprocessed_data.txt', 'w') as f:
    for pair in input_output_pairs:
        f.write(pair[0] + '\t' + pair[1] + '\n')
