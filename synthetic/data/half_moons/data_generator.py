from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=250, noise=0.125, random_state=42)

colormap = {1: 'tab:blue', 0: 'tab:red'}
for i,x in enumerate(X):
    plt.scatter(x[0], x[1], marker='o', color=colormap[y[i]])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.savefig('dist.png', dpi=300, bbox_inches='tight')



y = ["Y" if el == 0 else "N" for el in y]
y = np.array(y).reshape((-1,1))

s = []
for _ in y:
    s.append(np.random.choice(["Male", "Female"]))
s = np.array(s).reshape((-1,1))

X = np.append(X, s, 1)
X = np.append(X, y, 1)

X_df = pd.DataFrame(data=X, columns=None)
print(X_df)

X_df.to_csv('train.csv', index=False, header=None)
#X_df.to_csv('test.csv', index=False, header=None) #This will now be handled by test_generator.py
