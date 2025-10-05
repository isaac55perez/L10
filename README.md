# Association Rule Mining Project

This project implements the Apriori algorithm to discover association rules in binary data. It generates synthetic data with known patterns and applies association rule mining to find interesting relationships.

## Description

The program performs two main tasks:
1. Generates a synthetic dataset with built-in associations
2. Discovers association rules using the Apriori algorithm

### Data Generation
Creates a table with 5000 rows and 6 columns (A-F), where each cell contains either 0 or 1. The data includes:

**Base Probabilities:**
- Column A: 70% probability of 1
- Column B: 30% probability of 1
- Column C: 60% probability of 1
- Columns D, E, F: Base 50% probability of 1

**Built-in Associations:**
1. When A and C are both 1, D has 80% chance of being 1
2. When B is 1, E has 90% chance of being 1
3. When C and D are both 1, F has 85% chance of being 1

## Requirements

- Python 3.6 or higher
- NumPy library

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source .venv/bin/activate
     ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the program using Python:
```bash
python association.py
```

The program will:
1. Generate the random data with built-in associations
2. Display data samples and column distributions
3. Find association rules meeting the thresholds:
   - Minimum support: 30%
   - Minimum confidence: 70%

## Results

### Data Distribution
```
Actual proportions of 1s in each column:
Column A: 69.38%
Column B: 28.34%
Column C: 60.14%
Column D: 61.60%
Column E: 62.40%
Column F: 64.94%
```

### Discovered Association Rules
The algorithm found 17 significant rules. Here are the top findings (sorted by confidence):

1. Strong Association Rules (Confidence > 80%):
   ```
   C AND D => F (support: 35.96%, confidence: 84.93%, lift: 1.31)
   ```
   - When items C and D appear together, F appears 84.93% of the time
   - This relationship occurs in 35.96% of all transactions
   - F is 1.31 times more likely when C and D are present

2. Notable Rules (Confidence > 75%):
   ```
   C AND F => D (support: 35.96%, confidence: 79.66%, lift: 1.29)
   A AND C => D (support: 32.84%, confidence: 79.29%, lift: 1.29)
   F AND D => C (support: 35.96%, confidence: 79.24%, lift: 1.32)
   A AND C => F (support: 32.24%, confidence: 77.84%, lift: 1.20)
   C AND D => A (support: 32.84%, confidence: 77.56%, lift: 1.12)
   ```

### Interesting Findings

1. **Successfully Detected Associations:**
   - The algorithm found the built-in C AND D => F relationship
   - The confidence (84.93%) closely matches our built-in 85% probability

2. **Missing Associations:**
   - B => E relationship (90% confidence) wasn't found because B's support (28.34%) is below threshold
   - This demonstrates how minimum support can hide strong but rare relationships

3. **Unexpected Patterns:**
   - Several three-item relationships emerged
   - All rules show positive lift (> 1), indicating genuine correlations

## Algorithm Metrics

The implementation uses three key metrics:

1. **Support:**
   - Frequency of an itemset in the data
   - Used to eliminate rare combinations
   - Threshold: 30%

2. **Confidence:**
   - Probability of finding the consequent given the antecedent
   - Measures rule reliability
   - Threshold: 70%

3. **Lift:**
   - Ratio of observed support to expected support
   - Measures how much the rule improves prediction
   - All discovered rules have lift > 1

## License

This project is open source and available under the MIT License.