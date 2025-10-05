import numpy as np

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Create a table with 5000 rows and 6 columns
rows = 5000
cols = 6

# Initialize an empty array
data = np.zeros((rows, cols), dtype=int)

# Fill columns with random values based on specified probabilities
# Column A: 70% probability of 1
data[:, 0] = np.random.choice([0, 1], size=rows, p=[0.3, 0.7])

# Column B: 30% probability of 1
data[:, 1] = np.random.choice([0, 1], size=rows, p=[0.7, 0.3])

# Column C: 60% probability of 1
data[:, 2] = np.random.choice([0, 1], size=rows, p=[0.4, 0.6])

# Columns D, E, F: 50% probability of 1
data[:, 3:] = np.random.choice([0, 1], size=(rows, 3), p=[0.5, 0.5])

# Create column labels
column_labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Print first few rows as a sample
print("Sample of the generated data (first 10 rows):")
print("Columns:", column_labels)
print(data[:10])

# Print statistics to verify probabilities
print("\nActual proportions of 1s in each column:")
for col, label in enumerate(column_labels):
    proportion = np.mean(data[:, col])
    print(f"Column {label}: {proportion:.2%}")
