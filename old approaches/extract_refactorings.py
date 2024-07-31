import json
import os
import subprocess
from pathlib import Path

# Function to run shell commands
def run_command(command):
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    return result.stdout.strip()

# Function to extract lines of code from a file
def extract_lines(file_path, lines):
    with open(file_path, 'r') as file:
        code = file.readlines()
    return ''.join([code[line - 1] for line in lines])

# Path to the JSON file
json_file_path = 'flask_repo_filtered.json'

# Load JSON data
with open(json_file_path, 'r') as file:
    refactorings = json.load(file)

# Clone the repository (replace with the actual repository URL)
repo_url = 'https://github.com/pallets/flask.git'
repo_dir = 'flask_repo'

# Clone the repository if not already cloned
if not Path(repo_dir).exists():
    run_command(f'git clone {repo_url} {repo_dir}')

# Change to the repository directory
os.chdir(repo_dir)

# Directory to store before and after code snippets
output_dir = Path('../refactorings_output')
output_dir.mkdir(exist_ok=True)

dataset = []

# Process each refactoring
for refactoring in refactorings:
    commit = refactoring['Commit']
    location_parts = refactoring['Location'].split('/')
    file_path = '/'.join(location_parts[:-1])
    file_name = location_parts[-1]
    full_file_path = os.path.join(file_path, file_name)
    
    original_lines = list(range(int(refactoring['Original Method Line'][1:-1].split(',')[0]), int(refactoring['Original Method Line'][1:-1].split(',')[1]) + 1))
    extracted_lines = [int(line) for line in refactoring['Extracted/Inlined Lines']]

    # Checkout the commit
    run_command(f'git checkout {commit}')

    # Extract original method code
    original_code = extract_lines(full_file_path, original_lines)

    # Extract extracted method code
    extracted_code = extract_lines(full_file_path, extracted_lines)

    # Save the code snippets to files
    original_code_file = output_dir / f"{commit}_original_{refactoring['Original']}.txt"
    extracted_code_file = output_dir / f"{commit}_extracted_{refactoring['Updated']}.txt"
    
    with open(original_code_file, 'w') as file:
        file.write(original_code)
    
    with open(extracted_code_file, 'w') as file:
        file.write(extracted_code)

    dataset.append({
        'Commit': commit,
        'Original Method': refactoring['Original'],
        'Extracted Method': refactoring['Updated'],
        'Original Code File': str(original_code_file),
        'Extracted Code File': str(extracted_code_file),
        'Description': refactoring['Description']
    })

# Save the dataset to a JSON file
dataset_file = 'refactoring_dataset.json'
with open(dataset_file, 'w') as file:
    json.dump(dataset, file, indent=4)

print(f"Dataset saved to {dataset_file}")
