# -*- coding: utf-8 -*-


class NoHubnessReduction:
    """ Dummy class to test the library """

    def fit(self, neigh_dist=None, neigh_ind=None):
        pass

    def transform(self, neigh_dist=None, neigh_ind=None, return_distance=True, *args, **kwargs):
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind

    def fit_transform(self, neigh_dist=None, neigh_ind=None, return_distance=True, *args, **kwargs):
        self.fit(neigh_dist, neigh_ind)
        return self.transform(neigh_dist, neigh_ind, return_distance=return_distance)
