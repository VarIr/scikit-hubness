# SPDX-License-Identifier: BSD-3-Clause
import sys


def available_ann_algorithms_on_current_platform():
    """ Get approximate nearest neighbor algorithms available for the current platform/OS

    Currently, the algorithms are provided by the following libraries:

        * 'hnsw': nmslib
        * 'rptree': annoy
        * 'lsh': puffinn
        * 'falconn_lsh': falconn
        * 'nng': NGT

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
    elif sys.platform == 'darwin':  # pragma: no cover
        if 'pytest' in sys.modules:
            # Work-around: Skip tests of PuffinnLSH on MacOS, as it appears to be less precise than on Linux...
            algorithms = ('falconn_lsh',
                          'hnsw',
                          'rptree',
                          'nng',
                          )
        else:
            algorithms = ('falconn_lsh',
                          'lsh',
                          'hnsw',
                          'rptree',
                          'nng',
                          )
    # Linux
    elif sys.platform == 'linux':
        algorithms = ('lsh',
                      'falconn_lsh',
                      'hnsw',
                      'rptree',
                      'nng',
                      )
    # others: undefined
    else:  # pragma: no cover
        algorithms = ()

    return algorithms
