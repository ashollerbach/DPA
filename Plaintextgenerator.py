import random

# Set the output file path
output_file = "./plaintexts.txt"

# Generate 1000 random 128-bit plaintexts
random.seed(0)  # For reproducibility
plaintexts = []
for _ in range(1000):
    plaintext = ''.join(f"{random.randint(0, 255):02x}" for _ in range(16))  # 16 bytes = 128 bits
    for i in range(20):
        plaintexts.append(plaintext)
    

# Save plaintexts to a .txt file
with open(output_file, "w") as file:
    for plaintext in plaintexts:
        file.write(plaintext + "\n")

print(f"Generated 1000 plaintexts and saved to {output_file}")
