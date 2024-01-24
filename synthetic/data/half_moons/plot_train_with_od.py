import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv', header=None)

f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()

indices = np.load('od_idxs.npy').tolist()
print(indices)

y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

for i, (a,b) in enumerate(zip(f1, f2)):
    if i in indices:
        plt.scatter(a,b, marker='X', color=y_colormap[y[i]], s=75, edgecolor='black')
        continue

    plt.scatter(a,b, marker='o', color=y_colormap[y[i]])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.savefig('dist_od.png', dpi=300, bbox_inches='tight')

