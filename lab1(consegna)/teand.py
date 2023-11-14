import numpy as np

# Set a seed value for reproducibility


# Generate a single random number between 1 and 10 (inclusive)
np.random.seed(12)  # Resetting the seed before each random number generation
for i in range(0,5): 
    print(np.random.randint(1, 11))  # Range: [1, 10])
print("secon \n\n")

np.random.seed(12)  # Resetting the seed before each random number generation
for i in range(0,5): 
    print(np.random.randint(1, 11))  # Range: [1, 10])
print("tri \n\n")


np.random.seed(13)  # Resetting the seed before each random number generation
for i in range(0,5): 
    print(np.random.randint(1, 11))  # Range: [1, 10])

