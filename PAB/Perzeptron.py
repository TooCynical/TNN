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
import random

import Transfer
import Random

from Parsing import parse_training_file


N_MAX = 101
M_MAX = 30
T_MAX = 201


class PerzLayer:
    def __init__(self, N=1, M=1, learning_factor=0.0001, transfer=Transfer.identity_transfer, sampler=Random.uniform_sampler(-2, 2)):
        self.__init_transfer(transfer)
        self.__init_learning_factor(learning_factor)
        self.__init_sampler(sampler)
        self.__init_weights_from_dimensions(N, M)
        self.__init_pending_weights()


    # Evaluate the layer for given input by applying the transfer function
    # to each entry of W * [1 X]^T. Save the input and the returned vector 
    # within the layer until a next call is made for training.
    def __call__(self, X):
        self.net = (np.dot(self.W[:, 1:], X) + self.W[:, 0]).flatten()
        self.out = self.transfer_vec(self.net).flatten()
        return self.out


    # Set delta value for each neuron in this layer.    
    def set_deltas(self, deltas):
        self.deltas = deltas
        self.deltas.shape = (self.deltas.shape[0], 1)


    # Apply the stored weight changes from learning, and reset them.
    def apply_pending_weights(self):
        self.W += self.pending_weights
        self.__init_pending_weights()


    def update_pending_weights(self, new_pending_weights):
        self.pending_weights += new_pending_weights


    # Initialize weights as an Mx(N+1) matrix of values picked at random.
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


    # Create a vectorized version of the transfer function and its derivative.
    def __init_transfer(self, transfer):                
        if not isinstance(transfer, Transfer.TransferFunction):
            raise TypeError("Transfer function should be a TransferFunction object!")
        self.transfer_vec = np.vectorize(transfer.f)
        self.transfer_prime_vec = np.vectorize(transfer.f_prime)


    def __init_pending_weights(self):
        self.pending_weights = np.zeros((self.M, self.N + 1))

    
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
        return X


    # Given a list [N1, N2, N3, ...] of dimensions, a list of learning rates,
    # and a list of transfer functions, create layers N1 -> N2 -> ... with
    # corresponding learning rates and transfer functions.
    def __init_layers_from_dimensions(self, dims, etas, transfers):
        self.layers = []
        for i in xrange(1, len(dims)):
            self.layers.append(PerzLayer(dims[i-1], dims[i]))

   # Train the Perzeptron once given a list of patterns using back 
   # propagation and the Widrow-Hoff rule.
    def train(self, patterns):
        # First gather all weight changes
        for P in patterns:
            if P.N == self.layers[0].N and P.M == self.layers[-1].M:
                self.__train_pattern(P)
            else:
                raise ValueError("Training input dimension mismatch. \n")

        # Apply weight changes only after all patterns have been applied.
        for layer in self.layers:
            layer.apply_pending_weights()


    # Train the Perzeptron with a single pattern using back propagation and 
    # the Widrow-Hoff training rule. Does not apply weight changes!
    def __train_pattern(self, P):
        Y = self(P.X)
        self.__update_out_layer(P, Y)
        for i in xrange(1, len(self.layers)):
            self.__update_hidden_layer(P, Y, i)

    # Find weight changes for the output layer.
    def __update_out_layer(self, P, Y):
        out_layer = self.layers[-1]

        if (len(self.layers) > 1):
            prev_out = self.layers[-2].out
        else:
            prev_out = P.X
        prev_out = np.insert(prev_out, 0, 1.0)
        prev_out.shape = (1, prev_out.shape[0])

        # Compute f'(net_m) for all values of m as a vector for output layer.
        fprime_net = out_layer.transfer_prime_vec(out_layer.net)

        # Set \delta_m for all values of m as a vector for output layer.
        out_layer.set_deltas(np.multiply(P.Y - Y, fprime_net))

        # Update delta_weights
        new_delta_weights = out_layer.learning_factor * \
                            (np.dot(out_layer.deltas, prev_out))
        out_layer.update_pending_weights(new_delta_weights)

    # Find weight changes for the ith hidden layer (counting 
    # backwards from the output layer).
    def __update_hidden_layer(self, P, Y, i):
        cur_layer = self.layers[-1 - i]
        next_layer = self.layers[-i]

        if (len(self.layers) > 1 + i):
            prev_out = self.layers[-2 - i].out
        else:
            prev_out = P.X
        prev_out = np.insert(prev_out, 0, 1.0)
        prev_out.shape = (1, prev_out.shape[0])

        # Compute f'(net_m) for all values of m as a vector for output layer.
        fprime_net = cur_layer.transfer_prime_vec(cur_layer.net)

        # Compute (sum_k w_hk delta_k)_h for all h as vector
        delta_bar = next_layer.deltas

        W_bar = np.transpose(next_layer.W[:, 1:])
        cur_layer.set_deltas(np.multiply(np.dot(W_bar, delta_bar).flatten(), fprime_net)) 

        # Update delta_weights
        new_delta_weights = cur_layer.learning_factor * \
                            (np.matmul(cur_layer.deltas, prev_out))
        cur_layer.update_pending_weights(new_delta_weights)
        

    def verify(self, P):
        return quadratic_error(self(P.X), P.Y)


    def __repr__(self):
        out = ""
        for layer in self.layers:
            out += str(layer) + "\n"
        return out


# Return half of the quadratic distance X - Y.
def quadratic_error(X, Y):
    return 0.5 * np.sum((X - Y)**2)


# A training pattern consisting of an input vector and a desired output vector.
class Pattern:
    # Construct a pattern from input and desired output.
    def __init__(self, X, Y):
        if X.ndim == 1 and Y.ndim == 1:
            self.X = X
            self.Y = Y
        
            self.N = X.shape[0]
            self.M = Y.shape[0]
        
        else:
            raise ValueError("pattern should be constructed from two vectors. \n")
    
    
    # Return whether the input and output of this pattern are binary.
    def is_binary(self):
        return ((self.X == 0) | (self.X == 1)).all() and ((self.Y == 0) | (self.Y == 1)).all()


if __name__ == "__main__":
    q = Perzeptron([2, 3, 4, 4, 10, 4])

    X = np.array([0, 1])
    Y = np.array([3, 2, 1, 0])
    P = Pattern(X, Y)
    print q.verify(P)

    for i in range(200):
        q.train([P])
        print q.verify(P)

