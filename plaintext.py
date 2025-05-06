import os
import random

def generate_hex_file(file_path, hex_value, rows=1000):
    """
    Generates a file filled with a specified hex value repeated for a given number of rows.

    Args:
        file_path (str): Path to the file to be created.
        hex_value (str): The 128-bit hex value to write to the file.
        rows (int): Number of rows to write.
    """
    with open(file_path, 'w') as file:
        for _ in range(rows):
            file.write(f"{hex_value}\n")

# Create the output directory
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# Generate 1000 random hex files
for i in range(1000):
    hex_value = ''.join(random.choices('0123456789abcdef', k=32))
    file_path = os.path.join(output_dir, f"{hex_value}.txt")
    generate_hex_file(file_path, hex_value)
    print(f"File {i + 1} generated at: {file_path}")
