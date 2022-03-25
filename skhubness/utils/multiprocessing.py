# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
from multiprocessing import cpu_count

__all__ = [
    "validate_n_jobs",
]


def register_parallel_pytest_cov():
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()


def validate_n_jobs(n_jobs):
    """ Handle special integers and non-integer `n_jobs` values. """
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"Number of parallel processes 'n_jobs' must be "
                         f"a positive integer, or ``-1`` to use all local"
                         f" CPU cores. Was {n_jobs} instead.")
    return n_jobs
