import pandas as pd
dataset = pd.read_csv("data.csv", header=None)

transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
  
from apyori import apriori
associations = apriori(transactions, min_support = 0.004, min_confidence = 0.4, min_lift = 3, min_length = 2)

print(list(associations))