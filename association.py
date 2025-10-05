import numpy as np
from itertools import combinations
from typing import List, Set, Dict, Tuple

class AprioriAssociation:
    """
    Apriori Association Rule Mining Algorithm
    
    Time Complexity Analysis:
    ------------------------
    Let:
    - n = number of transactions
    - m = number of items (features)
    - k = size of largest frequent itemset
    
    1. Finding frequent 1-itemsets: O(n*m)
       - Must scan each transaction for each item
    
    2. Generating candidate k-itemsets: O(l^2 * k)
       where l is the number of frequent (k-1)-itemsets
       - Requires comparing each pair of (k-1)-itemsets
    
    3. Counting support for candidates: O(n*c*k)
       where c is the number of candidate k-itemsets
       - Must scan each transaction for each candidate
    
    4. Rule generation: O(2^k * k)
       - For each frequent itemset of size k, must generate all possible subsets
    
    Overall worst-case complexity: O(n*m + 2^m)
    - The 2^m term comes from the potential number of itemsets
    - In practice, the min_support threshold greatly reduces this
    
    Space Complexity: O(2^m)
    - Must store frequent itemsets and candidate itemsets
    - Again, min_support reduces this significantly in practice
    """
    
    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.7):
        """
        Initialize the Apriori Association Rule Mining algorithm.
        
        Args:
            min_support (float): Minimum support threshold (default: 0.3)
            min_confidence (float): Minimum confidence threshold (default: 0.7)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        
    def _get_frequent_1_itemsets(self, data: np.ndarray, features: List[str]) -> Dict[frozenset, float]:
        """
        Find frequent 1-itemsets that meet the minimum support threshold.
        
        Args:
            data (np.ndarray): Binary transaction data
            features (List[str]): List of feature names
            
        Returns:
            Dict[frozenset, float]: Dictionary of frequent 1-itemsets and their support values
        """
        frequent_1_itemsets = {}
        n_transactions = len(data)
        
        for i, feature in enumerate(features):
            support = np.mean(data[:, i])
            if support >= self.min_support:
                frequent_1_itemsets[frozenset([feature])] = support
                
        return frequent_1_itemsets
    
    def _generate_candidates(self, prev_frequent_sets: Dict[frozenset, float], k: int) -> Set[frozenset]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets.
        
        Args:
            prev_frequent_sets (Dict[frozenset, float]): Previous frequent itemsets
            k (int): Size of itemsets to generate
            
        Returns:
            Set[frozenset]: Set of candidate k-itemsets
        """
        candidates = set()
        prev_frequent_items = list(prev_frequent_sets.keys())
        
        for i in range(len(prev_frequent_items)):
            for j in range(i + 1, len(prev_frequent_items)):
                items1 = prev_frequent_items[i]
                items2 = prev_frequent_items[j]
                
                # If first k-2 items are the same, merge the sets
                if len(items1.union(items2)) == k:
                    candidates.add(items1.union(items2))
                    
        return candidates
    
    def _calculate_support(self, itemset: frozenset, data: np.ndarray, features: List[str]) -> float:
        """
        Calculate support for an itemset.
        
        Args:
            itemset (frozenset): Set of items
            data (np.ndarray): Binary transaction data
            features (List[str]): List of feature names
            
        Returns:
            float: Support value
        """
        # Get column indices for the items in the itemset
        indices = [features.index(item) for item in itemset]
        
        # Check which transactions contain all items in the itemset
        transactions_with_items = np.all(data[:, indices] == 1, axis=1)
        
        return np.mean(transactions_with_items)
    
    def _generate_rules(self, frequent_itemsets: Dict[frozenset, float], data: np.ndarray, 
                       features: List[str]) -> List[Tuple[frozenset, frozenset, float, float, float]]:
        """
        Generate association rules that meet the minimum confidence threshold.
        
        Args:
            frequent_itemsets (Dict[frozenset, float]): Frequent itemsets and their support values
            data (np.ndarray): Binary transaction data
            features (List[str]): List of feature names
            
        Returns:
            List[Tuple[frozenset, frozenset, float, float, float]]: List of rules 
            (antecedent, consequent, support, confidence, lift)
        """
        rules = []
        
        for itemset in frequent_itemsets:
            if len(itemset) < 2:
                continue
                
            # Generate all possible antecedent-consequent pairs
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # Calculate confidence
                    antecedent_support = self._calculate_support(antecedent, data, features)
                    consequent_support = self._calculate_support(consequent, data, features)
                    confidence = frequent_itemsets[itemset] / antecedent_support
                    
                    # Calculate lift
                    lift = confidence / consequent_support
                    
                    if confidence >= self.min_confidence:
                        rules.append((antecedent, consequent, frequent_itemsets[itemset], 
                                   confidence, lift))
        
        return rules
    
    def fit(self, data: np.ndarray, features: List[str]) -> List[Tuple[frozenset, frozenset, float, float]]:
        """
        Find association rules in the data.
        
        Args:
            data (np.ndarray): Binary transaction data
            features (List[str]): List of feature names
            
        Returns:
            List[Tuple[frozenset, frozenset, float, float]]: List of rules (antecedent, consequent, support, confidence)
        """
        # Find frequent 1-itemsets
        frequent_itemsets = self._get_frequent_1_itemsets(data, features)
        k = 2
        
        # Find frequent k-itemsets
        while True:
            candidates = self._generate_candidates(frequent_itemsets, k)
            if not candidates:
                break
                
            # Calculate support for candidates
            new_frequent_itemsets = {}
            for candidate in candidates:
                support = self._calculate_support(candidate, data, features)
                if support >= self.min_support:
                    new_frequent_itemsets[candidate] = support
            
            if not new_frequent_itemsets:
                break
                
            frequent_itemsets.update(new_frequent_itemsets)
            k += 1
        
        # Generate rules from frequent itemsets
        return self._generate_rules(frequent_itemsets, data, features)

# Generate random data
def generate_data(rows: int = 5000) -> Tuple[np.ndarray, List[str]]:
    """
    Generate binary data with specified probabilities and some built-in associations.
    
    Args:
        rows (int): Number of rows to generate
        
    Returns:
        Tuple[np.ndarray, List[str]]: Generated data and feature names
    """
    np.random.seed(42)
    data = np.zeros((rows, 6), dtype=int)
    
    # Generate base probabilities
    data[:, 0] = np.random.choice([0, 1], size=rows, p=[0.3, 0.7])  # A: 70%
    data[:, 1] = np.random.choice([0, 1], size=rows, p=[0.7, 0.3])  # B: 30%
    data[:, 2] = np.random.choice([0, 1], size=rows, p=[0.4, 0.6])  # C: 60%
    
    # Create associations:
    # 1. When A and C are both 1, D is likely to be 1 (80% chance)
    # 2. When B is 1, E is likely to be 1 (90% chance)
    # 3. When C and D are both 1, F is likely to be 1 (85% chance)
    
    # Initialize D, E, F with base probability
    data[:, 3:] = np.random.choice([0, 1], size=(rows, 3), p=[0.5, 0.5])
    
    # Apply associations
    ac_mask = (data[:, 0] == 1) & (data[:, 2] == 1)
    data[ac_mask, 3] = np.random.choice([0, 1], size=np.sum(ac_mask), p=[0.2, 0.8])
    
    b_mask = (data[:, 1] == 1)
    data[b_mask, 4] = np.random.choice([0, 1], size=np.sum(b_mask), p=[0.1, 0.9])
    
    cd_mask = (data[:, 2] == 1) & (data[:, 3] == 1)
    data[cd_mask, 5] = np.random.choice([0, 1], size=np.sum(cd_mask), p=[0.15, 0.85])
    
    features = ['A', 'B', 'C', 'D', 'E', 'F']
    return data, features

def format_rule(rule: Tuple[frozenset, frozenset, float, float, float]) -> str:
    """
    Format an association rule for display.
    
    Args:
        rule (Tuple[frozenset, frozenset, float, float, float]): Association rule
        
    Returns:
        str: Formatted rule string
    """
    antecedent, consequent, support, confidence, lift = rule
    return (f"{' AND '.join(antecedent)} => {' AND '.join(consequent)} "
            f"(support: {support:.2%}, confidence: {confidence:.2%}, lift: {lift:.2f})")

def explain_metrics(data: np.ndarray, features: List[str]):
    """
    Explain support, confidence, and lift metrics using concrete examples from the data.
    """
    # Example 1: Single item support
    support_A = np.mean(data[:, 0])
    total_rows = len(data)
    rows_with_A = np.sum(data[:, 0])
    
    print("\nEXPLAINING ASSOCIATION RULE METRICS")
    print("===================================")
    print("\nExample 1: Single Item Support")
    print(f"Support for item A = {support_A:.2%}")
    print(f"This means that out of {total_rows} transactions:")
    print(f"- {rows_with_A} transactions contain A=1")
    print(f"- Support = {rows_with_A} / {total_rows} = {support_A:.2%}")
    
    # Example 2: Itemset support
    A_and_C = np.logical_and(data[:, 0] == 1, data[:, 2] == 1)
    support_AC = np.mean(A_and_C)
    rows_with_AC = np.sum(A_and_C)
    
    print("\nExample 2: Itemset Support")
    print(f"Support for itemset {{'A', 'C'}} = {support_AC:.2%}")
    print(f"This means that out of {total_rows} transactions:")
    print(f"- {rows_with_AC} transactions contain both A=1 AND C=1")
    print(f"- Support = {rows_with_AC} / {total_rows} = {support_AC:.2%}")
    
    # Example 3: Rule confidence and lift
    A_and_C_and_D = np.logical_and(A_and_C, data[:, 3] == 1)
    support_D = np.mean(data[:, 3])
    confidence_AC_D = np.sum(A_and_C_and_D) / rows_with_AC
    lift_AC_D = confidence_AC_D / support_D
    
    print("\nExample 3: Rule Confidence and Lift")
    print("Rule: IF A=1 AND C=1 THEN D=1")
    print(f"Confidence = {confidence_AC_D:.2%}")
    print("This means:")
    print(f"- Out of {rows_with_AC} transactions where A=1 AND C=1")
    print(f"- {np.sum(A_and_C_and_D)} transactions also have D=1")
    print(f"- Confidence = {np.sum(A_and_C_and_D)} / {rows_with_AC} = {confidence_AC_D:.2%}")
    print("- In other words, when we find A=1 AND C=1, there's a")
    print(f"  {confidence_AC_D:.2%} chance that D=1")
    
    print(f"\nLift = {lift_AC_D:.2f}")
    print("This means:")
    print(f"- The baseline probability of D=1 is {support_D:.2%}")
    print(f"- When A=1 AND C=1, the probability increases to {confidence_AC_D:.2%}")
    print(f"- Lift = {confidence_AC_D:.2%} / {support_D:.2%} = {lift_AC_D:.2f}")
    print("- In other words, D is", end=" ")
    if lift_AC_D > 1:
        print(f"{lift_AC_D:.2f} times more likely when A=1 AND C=1")
    elif lift_AC_D < 1:
        print(f"{1/lift_AC_D:.2f} times less likely when A=1 AND C=1")
    else:
        print("independent of A AND C (no relationship)")

def main():
    # Generate data
    data, features = generate_data()
    
    print("Sample of the generated data (first 10 rows):")
    print("Columns:", features)
    print(data[:10])
    
    print("\nActual proportions of 1s in each column:")
    for col, label in enumerate(features):
        proportion = np.mean(data[:, col])
        print(f"Column {label}: {proportion:.2%}")
    
    # Explain support and confidence using examples
    explain_metrics(data, features)
    
    # Find association rules
    print("\nFinding association rules (support ≥ 30%, confidence ≥ 70%)...")
    apriori = AprioriAssociation(min_support=0.3, min_confidence=0.7)
    rules = apriori.fit(data, features)
    
    # Sort rules by confidence and then support
    rules.sort(key=lambda x: (-x[3], -x[2]))
    
    print(f"\nFound {len(rules)} rules:")
    for rule in rules:
        print(format_rule(rule))

if __name__ == "__main__":
    main()
