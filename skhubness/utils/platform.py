# SPDX-License-Identifier: BSD-3-Clause
import sys


def available_ann_algorithms_on_current_platform():
    """ Get approximate nearest neighbor algorithms available for the current platform/OS

    Currently, the algorithms are provided by the following libraries:

        * 'hnsw': nmslib
        * 'rptree': annoy
        * 'lsh': puffinn
        * 'falconn_lsh': falconn
        * 'onng': NGT

    Returns
    -------
    algorithms: Tuple[str]
        A tuple of available algorithms
    """
    # Windows
    if sys.platform == 'win32':  # pragma: no cover
        algorithms = ('hnsw',
                      'rptree',
                      )
    # MacOS
    elif sys.platform == 'darwin':
        algorithms = ('falconn_lsh',
                      'hnsw',
                      'rptree',
                      'onng',
                      )
    # Linux
    elif sys.platform == 'linux':
        algorithms = ('lsh',
                      'falconn_lsh',
                      'hnsw',
                      'rptree',
                      'onng',
                      )
    # others: undefined
    else:  # pragma: no cover
        algorithms = ()

    return algorithms
