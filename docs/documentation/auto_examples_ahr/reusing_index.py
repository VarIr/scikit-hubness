"""
========================================
Example: Reusing index structures
========================================

This example shows how to reuse index structures. If you want to first estimate hubness,
and then perform kNN, you can avoid recomputing the ANN index structure, which can be
costly.
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skhubness.analysis import Hubness
from skhubness.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=100_000,
                           n_features=500,
                           n_informative=400,
                           random_state=543)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.01,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=2346)

# Approximate hubness estimation: Creates LSH index and computes local scaling factors
hub = Hubness(k=10,
              return_value='robinhood',
              algorithm='falconn_lsh',
              hubness='ls',
              random_state=2345,
              shuffle_equal=False,
              verbose=1)
hub.fit(X_train)

robin_hood = hub.score(X_test)
print(f'Hubness (Robin Hood): {robin_hood}:.4f')
# 0.9060

# Approximate hubness reduction for classification: Reuse index & factors
knn = KNeighborsClassifier(n_neighbor=10,
                           algorithm='falconn_lsh',
                           hubness='ls',
                           n_jobs=1)

knn.fit(hub.nn_index_, y_train)  # REUSE INDEX HERE
acc = knn.score(X_test, y_test)
print(f'Test accuracy: {acc:.3f}')
# 0.959
