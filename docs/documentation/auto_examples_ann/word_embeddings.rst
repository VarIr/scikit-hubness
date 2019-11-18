.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_documentation_auto_examples_ann_word_embeddings.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_documentation_auto_examples_ann_word_embeddings.py:


=============================
Retrieving GLOVE word vectors
=============================

In this example we will retrieve similar words from
GLOVE embeddings with an ANNG graph.

Precomputed ground-truth nearest neighbors are available
from `ANN benchmarks <http://ann-benchmarks.com/index.html#datasets>`__.


.. code-block:: default


    # For this example, the `h5py` package is required in addition to the requirements of scikit-hubness.
    # You may install it from PyPI by the following command (if you're in an IPython/Jupyter environment):
    # !pip install h5py

    import numpy as np
    import h5py
    from skhubness.neighbors import NearestNeighbors

    # Download the dataset with the following command.
    # If the dataset is already available in the current working dir, you can skip this:
    # !wget http://ann-benchmarks.com/glove-100-angular.hdf5
    f = h5py.File('glove-100-angular.hdf5', 'r')

    # Extract the split and ground-truth
    X_train = f['train']
    X_test = f['test']
    neigh_true = f['neighbors']
    dist = f['distances']

    # How many object have we got?
    for k in f.keys():
        print(f'{k}: shape = {f[k].shape}')

    # APPROXIMATE NEAREST NEIGHBOR SEARCH
    # In order to retrieve most similar words from the GLOVE embeddings,
    # we use the unsupervised `skhubness.neighbors.NearestNeighbors` class.
    # The (approximate) nearest neighbor algorithm is set to NNG by passing `algorithm='nng'`.
    # We can pass additional parameters to `NNG` via the `algorithm_params` dict.
    # Here we set `n_jobs=8` to enable parallelism.
    # Create the nearest neighbor index
    nn_plain = NearestNeighbors(n_neighbors=100,
                                algorithm='nng',
                                algorithm_params={'n_candidates': 1_000,
                                                  'index_dir': 'auto',
                                                  'n_jobs': 8},
                                verbose=2,
                                )
    nn_plain.fit(X_train)

    # Note that NNG must save its index. By setting `index_dir='auto'`,
    # NNG will try to save it to shared memory, if available, otherwise to $TMP.
    # This index is NOT removed automatically, as one will typically want build an index once and use it often.
    # Retrieve nearest neighbors for each test object
    neigh_pred_plain = nn_plain.kneighbors(X_test,
                                           n_neighbors=100,
                                           return_distance=False)

    # Calculate the recall per test object
    recalled_plain = [np.intersect1d(neigh_true[i], neigh_pred_plain)
                      for i in range(len(X_test))]
    recall_plain = np.array([recalled_plain[i].size / neigh_true.shape[1]
                             for i in range(len(X_test))])

    # Statistics
    print(f'Mean = {recall_plain.mean():.4f}, '
          f'stdev = {recall_plain.std():.4f}')


    # ANN with HUBNESS REDUCTION
    # Here we set `n_candidates=1000`, so that for each query,
    # 1000 neighbors will be retrieved first by `NNG`,
    # that are subsequently refined by hubness reduction.
    # Hubness reduction is performed by local scaling as specified with `hubness='ls'`.
    # Creating the NN index with hubness reduction enabled
    nn = NearestNeighbors(n_neighbors=100,
                          algorithm='nng',
                          algorithm_params={'n_candidates': 1_000,
                                            'n_jobs': 8},
                          hubness='ls',
                          verbose=2,
                          )
    nn.fit(X_train)

    # Retrieve nearest neighbors for each test object
    neigh_pred = nn.kneighbors(X_test,
                               n_neighbors=100,
                               return_distance=False)

    # Measure recall per object and on average
    recalled = [np.intersect1d(neigh_true[i], neigh_pred)
                for i in range(len(X_test))]
    recall = np.array([recalled[i].size / neigh_true.shape[1]
                       for i in range(len(X_test))])
    print(f'Mean = {recall.mean():.4f}, '
          f'stdev = {recall.std():.4f}')

    # If the second results are significantly better than the first,
    # this could indicate that the chosen ANN method is more prone
    # to hubness than exact NN, which might be an interesting research question.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_documentation_auto_examples_ann_word_embeddings.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: word_embeddings.py <word_embeddings.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: word_embeddings.ipynb <word_embeddings.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
