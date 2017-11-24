import numpy as np
import itertools

import Util


# Singular SOM.
class SOM:
    class Center:
        def __init__(self, N, pos=None):
            if pos != None:
                self.pos = pos
            else:
                sampler = Util.uniform_sampler(0, 1)
                self.pos = [sampler() for dummy in range(N)]
            
            self.N = N


        def __call__(self, X):
            return Util.quadratic_error(self.pos, X)
    

        def __repr__(self):
            return str(self.N) + '-center @ ' + str(self.pos)

    def __init__(self, N=1, g=1, K=[1], F=[1], eta_0=1., eta_final=0., t_final=1000):
        if g != len(K) or g != len(F):
            raise ValueError("Grid dimension and length of K, F should match.")
        
        self.N = N
        self.g = g
        self.K = K
        self.F = F

        self.__init_centers()


    def __winner(self, X):
        min_distance = -1
        best_center = None
        for center in self.centers:
            new_distance = center(X)
            if new_distance < min_distance or min_distance == -1:
                min_distance = new_distance
                best_center = center
        return best_center


    # Returns the distance between two grid points given by their indices,
    # taking into account the edge weights.
    def __grid_distance(self, index1, index2):
        total_distance = 0
        for i in range(self.g):
            total_distance += abs(index1[i] - index2[i]) * self.F[i]
        return total_distance


    def __init_centers(self):
        l = [range(k) for k in self.K]
        self.grid_indices = []
        for index in itertools.product(*l):
            self.grid_indices.append(index)

        self.centers = {}
        for index in self.grid_indices:
            self.centers[index] = SOM.Center(self.N)


s = SOM(2, 3, [2,3,3], [10,10,10])

