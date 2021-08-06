# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod


class HubnessReduction(ABC):
    """ Base class for hubness reduction in a sparse neighbors graph. """
    @abstractmethod
    def __init__(self, **kwargs):
        # TODO whether to include/exclude self distances, or let the user decide...
        pass

    @abstractmethod
    def fit(self, X, y=None, **kwargs):
        pass

    @abstractmethod
    def transform(self, X, y=None, **kwargs):
        pass
