import pandas as pd
import os
import numpy as np

print(os.getcwd())
path = "./nlp/data/test.tsv"
df = pd.read_csv(path, sep='\t')



print(df.head())

a = [1,2,3,4,5,6,7]
b = []
b.append([x for x in a])

b = np.array(b)
b = b.flatten()

print(b)