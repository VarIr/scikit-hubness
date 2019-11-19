"""
========================================
Example: Approximate hubness reduction
========================================

This example shows how to combine approximate nearest neighbor search and hubness reduction
in order to perform approximate hubness reduction for large data sets.
"""
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skhubness.analysis import Hubness
from skhubness.neighbors import KNeighborsClassifier

# High-dimensional artificial data
X, y = make_classification(n_samples=1_000_000,
                           n_features=500,
                           n_informative=400,
                           random_state=543)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=10_000,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=2346)

# Approximate hubness estimation
hub = Hubness(k=10,
              return_value='robinhood',
              algorithm='hnsw',
              random_state=2345,
              shuffle_equal=False,
              n_jobs=-1,
              verbose=2)
hub.fit(X_train)
robin_hood = hub.score(X_test)
print(f'Hubness (Robin Hood): {robin_hood:.3f}')

# Approximate hubness reduction for classification
knn = KNeighborsClassifier(n_neighbor=10,
                           algorithm='hnsw',
                           hubness='ls',
                           n_jobs=-1,
                           verbose=2)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {acc:.3f}')
