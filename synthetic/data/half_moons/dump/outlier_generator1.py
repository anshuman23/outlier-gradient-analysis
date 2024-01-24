import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv', header=None)

f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()

idxs1 = []
idxs2 = []

for i, (f1_val, f2_val), in enumerate(zip(f1,f2)):
    if y[i] == 'Y':
        if f2_val >= 1.0 and f1_val <= 0.1 and f1_val >= -0.1:
            idxs1.append(i)
    else:
        if f2_val <= -0.5 and f1_val <= 1.4 and f1_val >= 1.0:
            idxs2.append(i)

print(idxs1, len(idxs1))
print(idxs2, len(idxs2))

idxs = idxs1 + idxs2

np.save('od_idxs.npy', idxs)

for i in idxs1:
    y[i] = 'N'
for i in idxs2:
    y[i] = 'Y'

df2 = pd.DataFrame({'a': f1, 'b': f2, 'c': df[2].to_list(), 'd': y}) #Header label doesn't matter, we won't save with it anyway
df2.to_csv('train.csv', header=None, index=False)
