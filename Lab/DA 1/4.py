import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    transactions = data['item'].apply(lambda x: x.split(', ')).tolist()
    return transactions

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

def run_fp_growth(df, min_support):
    return fpgrowth(df, min_support=min_support, use_colnames=True)

def generate_association_rules(frequent_itemsets, metric, min_threshold):
    return association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

print("Made by Anuj Parihar 21BBS0162")
file_path = 'monthly_sales.csv'
min_support = 0.2

transactions = load_and_preprocess_data(file_path)
df = encode_transactions(transactions)

frequent_itemsets = run_fp_growth(df, min_support)
print("Frequent Itemsets:")
print(frequent_itemsets)

rules = generate_association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
print("\nAssociation Rules:")
print(rules)

