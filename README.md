# Random Binary Data Generator

This project generates a table of random binary data (0s and 1s) using NumPy, with specific probability distributions for each column.

## Description

The program creates a table with 5000 rows and 6 columns (A-F), where each cell contains either 0 or 1. The probability of getting a 1 varies by column:

- Column A: 70% probability of 1
- Column B: 30% probability of 1
- Column C: 60% probability of 1
- Columns D, E, F: 50% probability of 1

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
1. Generate the random data table
2. Display the first 10 rows as a sample
3. Show the actual proportions of 1s in each column

## Output Example

```
Sample of the generated data (first 10 rows):
Columns: ['A', 'B', 'C', 'D', 'E', 'F']
[[1 0 0 0 1 1]
 [1 0 0 0 0 0]
 [1 1 0 0 1 1]
 ...]

Actual proportions of 1s in each column:
Column A: 69.38%
Column B: 28.34%
Column C: 60.14%
Column D: 49.82%
Column E: 50.56%
Column F: 51.56%
```

## License

This project is open source and available under the MIT License.