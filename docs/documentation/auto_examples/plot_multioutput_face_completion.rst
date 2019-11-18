.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_documentation_auto_examples_plot_multioutput_face_completion.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_documentation_auto_examples_plot_multioutput_face_completion.py:


===================================================
Face completion with a multi-output estimators
===================================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, linear regression, ridge regression,
and k nearest neighbors with or without hubness reduction
complete the lower half of those faces.


Adapted from `<https://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html>`_



.. image:: /documentation/auto_examples/images/sphx_glr_plot_multioutput_face_completion_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    /home/user/feldbauer/PycharmProjects/hubness/examples/sklearn/plot_multioutput_face_completion.py:106: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()





|


.. code-block:: default

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.utils.validation import check_random_state

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RidgeCV

    from skhubness.neighbors import KNeighborsRegressor

    # Load the faces datasets
    data = fetch_olivetti_faces()
    targets = data.target

    data = data.images.reshape((len(data.images), -1))
    train = data[targets < 30]
    test = data[targets >= 30]  # Test on independent people

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(4)
    face_ids = rng.randint(test.shape[0], size=(n_faces, ))
    test = test[face_ids, :]

    n_pixels = data.shape[1]
    # Upper half of the faces
    X_train = train[:, :(n_pixels + 1) // 2]
    # Lower half of the faces
    y_train = train[:, n_pixels // 2:]
    X_test = test[:, :(n_pixels + 1) // 2]
    y_test = test[:, n_pixels // 2:]

    # Fit estimators
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                           random_state=0),
        "k-NN": KNeighborsRegressor(weights='distance'),
        "k-NN MP": KNeighborsRegressor(hubness='mp',
                                       hubness_params={'method': 'normal'},
                                       weights='distance'),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }

    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    # Plot the completed faces
    image_shape = (64, 64)

    n_cols = 1 + len(ESTIMATORS)
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with multi-output estimators", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                              title="true faces")

        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                                  title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape),
                       cmap=plt.cm.gray,
                       interpolation="nearest")

    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.385 seconds)


.. _sphx_glr_download_documentation_auto_examples_plot_multioutput_face_completion.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_multioutput_face_completion.py <plot_multioutput_face_completion.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_multioutput_face_completion.ipynb <plot_multioutput_face_completion.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
