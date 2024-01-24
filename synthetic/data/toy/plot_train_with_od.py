import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', header=None)

f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()

df = pd.read_csv('idx_val.csv', header=None)
o1, o2 = df[0].to_list(), df[1].to_list()

indices = []
for a in o2:
    indices.append(f2.index(a))

print(indices)


y_colormap = {'Y': 'tab:blue', 'N': 'tab:red'}

for i, (a,b) in enumerate(zip(f1, f2)):
    if i in indices:
        plt.scatter(a,b, marker='X', color=y_colormap[y[i]], s=75, edgecolor='black')
        continue

    plt.scatter(a,b, marker='o', color=y_colormap[y[i]])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.savefig('dist.png', dpi=300, bbox_inches='tight')
