import pandas as pd
from itertools import combinations
from collections import Counter

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['item'] = data['item'].apply(lambda x: frozenset(x.split(',')))
    return data['item'].tolist()

def generate_candidates(prev_itemsets, k):
    return set(frozenset(i) for i in combinations(set().union(*prev_itemsets), k))

def get_frequent_itemsets(transactions, candidates, min_support):
    itemset_counts = Counter(
        frozenset(itemset) 
        for transaction in transactions 
        for itemset in candidates
        if set(itemset).issubset(transaction)
    )
    
    n_transactions = len(transactions)
    return {
        frozenset(itemset): count / n_transactions 
        for itemset, count in itemset_counts.items() 
        if count / n_transactions >= min_support
    }


def apriori(transactions, min_support):
    unique_items = set().union(*transactions)
    k = 1
    frequent_itemsets = {}

    while True:
        if k == 1:
            candidates = [{item} for item in unique_items]
        else:
            candidates = generate_candidates(frequent_itemsets[k-1].keys(), k)
        
        current_frequent = get_frequent_itemsets(transactions, candidates, min_support)
        
        if not current_frequent:
            break
        
        frequent_itemsets[k] = current_frequent
        k += 1

    return frequent_itemsets

def print_frequent_itemsets(frequent_itemsets):
    print("Frequent Itemsets:")
    for k, itemsets in frequent_itemsets.items():
        print(f"{k}-itemsets:")
        for itemset, support in itemsets.items():
            print(f"  {set(itemset)}: {support:.2f}")

file_path = 'juice_transactions.csv'
min_support = 0.2
print("Made by Anuj Parihar 21BBS0162")

transactions = load_and_preprocess_data(file_path)
frequent_itemsets = apriori(transactions, min_support)
print_frequent_itemsets(frequent_itemsets)
