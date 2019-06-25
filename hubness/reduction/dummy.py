# -*- coding: utf-8 -*-


class NoHubnessReduction:
    """ Dummy class to test the library """


    def fit(self, neigh_dist, neigh_ind):
        pass

    def transform(self, neigh_dist, neigh_ind, *args, **kwargs):
        return neigh_dist, neigh_ind

    def fit_transform(self, neigh_dist, neigh_ind):
        self.fit(neigh_dist, neigh_ind)
        return self.transform(neigh_dist, neigh_ind)
