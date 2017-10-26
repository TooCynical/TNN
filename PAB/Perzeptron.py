# Lucas S.   	lfh.slot@gmail.com
# Julian W.    	jjwarg@gmail.com  
#
# University of Bonn
# October 2017
#
# Perzepetron.py
# Implementation of a perzeptron without hidden layers with learning.


import numpy as np
import re

import Transfer
import Random

from Parsing import parse_training_file


N_MAX = 101
M_MAX = 30
T_MAX = 201

class PerzLayer:
    def __init__(self, N=1, M=1, learning_factor=0.1, transfer=Transfer.identity_transfer, sampler=Random.uniform_sampler(-2, 2)):
        self.__init_transfer(transfer)
        self.__init_learning_factor(learning_factor)
        self.__init_sampler(sampler)
        self.__init_weights_from_dimensions(N, M)
        

    # Evaluate the layer for given input by applying the transfer function
    # to each entry of W * [1 X]^T.
    def __call__(self, X):
        return self.transfer_vec(np.dot(self.W[:, 1:], X) + self.W[:, 0]).flatten()


    # Initialize weights as an NxM matrix of values picked at random.
    def __init_weights_from_dimensions(self, N, M):
        if N <= N_MAX and M <= M_MAX:
            self.N = N
            self.M = M
            self.W = np.array([[self.sampler() for x in xrange(N + 1)] for y in xrange(M)])
        else:
            raise ValueError("Perzeptron dimensions exceed bounds.")


    def __init_sampler(self, sampler):
        self.sampler = sampler


    def __init_learning_factor(self, learning_factor):
        self.learning_factor = learning_factor


    def __init_transfer(self, transfer):                
        if not isinstance(transfer, Transfer.TransferFunction):
            raise TypeError("Transfer function should be a TransferFunction object!")
        self.transfer_vec = np.vectorize(transfer.f)

    
    def __repr__(self):
        return "Layer (" + str(self.N) + "->" + str(self.M) \
                + "), learning factor: " + str(self.learning_factor) \
                + ",\nWeights (first column is BIAS): \n" + str(self.W)

class Perzeptron:
    def __init__(self, dims=[1, 1], etas=None, transfers=None):
        self.__init_layers_from_dimensions(dims, etas, transfers)


    def __call__(self, X):
        for L in self.layers:
            X = L(X)
            print L, X
        return X


    def __init_layers_from_dimensions(self, dims, etas, transfers):
        self.layers = []
        # Create input layer and hidden layers
        for i in xrange(1, len(dims)):
            self.layers.append(PerzLayer(dims[i-1], dims[i]))


    def __repr__(self):
        out = ""
        for layer in self.layers:
            out += str(layer) + "\n"
        return out


if __name__ == "__main__":
    q = Perzeptron([2,3,4, 1])
    X = np.array([2, 3])
    print q(X)
    print q
