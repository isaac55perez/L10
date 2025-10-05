# Association Rule Mining Project - Chat Log

## Initial Data Generation
Created a program to generate random binary data with specific probabilities:
- Column A: 70% probability of 1
- Column B: 30% probability of 1
- Column C: 60% probability of 1
- Columns D, E, F: 50% probability of 1

## Association Rule Mining Implementation
Implemented the Apriori algorithm with the following metrics:

### 1. Support
- Measures how frequently an itemset appears in the dataset
- Formula: support(X) = (transactions containing X) / (total transactions)
- Minimum threshold set to 30%

Example:
```
Support(A) = 69.38%
- Out of 5000 transactions, 3469 contain A=1
- This means item A appears in 69.38% of all transactions
```

### 2. Confidence
- Measures how likely rule's consequent is to appear when the antecedent appears
- Formula: confidence(X→Y) = support(X∪Y) / support(X)
- Minimum threshold set to 70%

Example:
```
Rule: IF A=1 AND C=1 THEN D=1
Confidence = 79.29%
- Out of 2071 transactions where A=1 AND C=1
- 1642 transactions also have D=1
- When we see A=1 AND C=1, there's a 79.29% chance that D=1
```

### 3. Lift
- Measures how much more likely the consequent is to occur when the antecedent is present
- Formula: lift(X→Y) = confidence(X→Y) / support(Y)
- Interpretation:
  - Lift > 1: Positive correlation
  - Lift = 1: No correlation
  - Lift < 1: Negative correlation

Example:
```
Rule: C AND D => F
Support: 35.96%
Confidence: 84.93%
Lift: 1.31 (F is 1.31 times more likely when C and D are present)
```

## Algorithm Complexity Analysis

### Time Complexity: O(n*m + 2^m)
Where:
- n = number of transactions
- m = number of items (features)
- k = size of largest frequent itemset

Breakdown:
1. Finding frequent 1-itemsets: O(n*m)
   - Scan each transaction for each item

2. Generating candidate k-itemsets: O(l^2 * k)
   - l = number of frequent (k-1)-itemsets
   - Compare each pair of (k-1)-itemsets

3. Counting support for candidates: O(n*c*k)
   - c = number of candidate k-itemsets
   - Scan each transaction for each candidate

4. Rule generation: O(2^k * k)
   - Generate all possible subsets for each frequent itemset

### Space Complexity: O(2^m)
- Storage needed for frequent itemsets and candidates
- Reduced significantly by minimum support threshold

### Practical Performance
Despite exponential theoretical complexity, the algorithm performs well in practice due to:
1. Support threshold eliminating many itemsets early
2. Real datasets rarely having frequent itemsets of large size
3. Most item combinations never becoming candidates

## Built-in Associations in Test Data
Created the following associations in the generated data:
1. When A and C are both 1, D has 80% chance of being 1
2. When B is 1, E has 90% chance of being 1
3. When C and D are both 1, F has 85% chance of being 1

## Found Association Rules
The strongest associations found (sorted by confidence):
1. C AND D => F (support: 35.96%, confidence: 84.93%, lift: 1.31)
2. C AND F => D (support: 35.96%, confidence: 79.66%, lift: 1.29)
3. A AND C => D (support: 32.84%, confidence: 79.29%, lift: 1.29)

Note: B's associations with E, despite having 90% confidence when B is present, don't appear in the results because B's support (28.34%) is below our minimum threshold of 30%.

## Project Structure
```
L10/
├── association.py     # Main implementation
├── requirements.txt   # Dependencies (numpy)
├── README.md         # Project documentation
├── LICENSE           # MIT License
├── .gitignore       # Git ignore rules
└── setup.py         # Package setup file
```

## Future Improvements
Potential optimizations for larger datasets:
1. Parallel processing for support counting
2. Hash-based techniques for candidate generation
3. Bit vectors for faster set operations
4. Consider implementing FP-Growth algorithm instead