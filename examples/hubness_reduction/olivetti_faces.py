"""
=================================
Face recognition (Olivetti faces)
=================================

This dataset contains a set of face images taken between April 1992
and April 1994 at AT&T Laboratories Cambridge.
Image data is typically embedded in very high-dimensional spaces,
which might be prone to hubness.
"""
import numpy as np
from sklearn.datasets import olivetti_faces
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV

from skhubness import Hubness
from skhubness.neighbors import KNeighborsClassifier

# Fetch data and have a look
d = olivetti_faces.fetch_olivetti_faces()
X, y = d['data'], d['target']
print(f'Data shape: {X.shape}')
print(f'Label shape: {y.shape}')
# (400, 4096)
# (400,)

# The data is embedded in a high-dimensional space.
# Is there hubness, and can we reduce it?
for hubness in [None, 'dsl', 'ls', 'mp']:
    hub = Hubness(k=10, hubness=hubness, return_value='k_skewness')
    hub.fit(X)
    score = hub.score()
    print(f'Hubness (10-skew): {score:.3f} with hubness reduction: {hubness}')
# Hubness (10-skew): 1.972 with hubness reduction: None
# Hubness (10-skew): 1.526 with hubness reduction: dsl
# Hubness (10-skew): 0.943 with hubness reduction: ls
# Hubness (10-skew): 0.184 with hubness reduction: mp

# There is some hubness, and all hubness reduction methods can reduce it (to varying degree)
# Let's assess the best kNN strategy and its estimated performance.
cv_perf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7263)
cv_select = StratifiedKFold(n_splits=5, shuffle=True, random_state=32634)

knn = KNeighborsClassifier(algorithm_params={'n_candidates': 100})

# specify parameters and distributions to sample from
param_dist = {"n_neighbors": np.arange(1, 26),
              "weights": ['uniform', 'distance'],
              "hubness": [None, 'dsl', 'ls', 'mp']}

# Inner cross-validation to select best hyperparameters (incl hubness reduction method)
search = RandomizedSearchCV(estimator=knn,
                            param_distributions=param_dist,
                            n_iter=100,
                            cv=cv_select,
                            random_state=2345,
                            verbose=1)

# Outer cross-validation to estimate performance
score = cross_val_score(search, X, y, cv=cv_perf, verbose=1)
print(f'Scores: {score}')
print(f'Mean acc = {score.mean():.3f} +/- {score.std():.3f}')

# Select model that maximizes accuracy
search.fit(X, y)

# The best model's parameters
print(search.best_params_)

# Does it correspond to the results of hubness reduction above?
# Scores: [0.95   0.9625 1.     0.95   0.925 ]
# Mean acc = 0.957 +/- 0.024
# {'weights': 'distance', 'n_neighbors': 23, 'hubness': 'mp'}
