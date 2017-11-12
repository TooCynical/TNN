# Lucas S.   	lfh.slot@gmail.com
# Julian W.    	jjwarg@gmail.com  
#
# University of Bonn
# November 2017
#
# Perzepetron.py
# Implementation of a perzeptron with hidden layers and learning.


import numpy as np
import re
import random

import Util

N_MAX = 1001
M_MAX = 30
T_MAX = 1001


class PerzLayer:
    def __init__(self, N=1, M=1, learning_factor=0.05, transfer=Util.tanh_transfer, sampler=Util.uniform_sampler(-2, 2)):
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


    # Set transfer function.
    def set_transfer_function(transfer):
        self.__init_transfer(transfer)


    # Set learning_factor.
    def set_learning_factor(learning_factor):
        self.__init_learning_factor(learning_factor)


    # Set delta value for each neuron in this layer.    
    def set_deltas(self, deltas):
        self.deltas = deltas
        self.deltas.shape = (self.deltas.shape[0], 1)


    # Apply the stored weight changes from learning, and reset them.
    def apply_pending_weights(self):
        self.W += self.pending_weights
        self.__init_pending_weights()


    # Increment pending weights.
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
        if not isinstance(transfer, Util.TransferFunction):
            raise TypeError("Tranfer function should be a TransferFunction object!")
        self.transfer_vec = np.vectorize(transfer.f)
        self.transfer_prime_vec = np.vectorize(transfer.f_prime)


    def __init_pending_weights(self):
        self.pending_weights = np.zeros((self.M, self.N + 1))

    
    def __repr__(self):
        return "Layer (" + str(self.N) + "->" + str(self.M) \
                + "), learning factor: " + str(self.learning_factor) \
                + ",\nWeights (first column is BIAS): \n" + str(self.W)


class Perzeptron:
    def __init__(self, dims=[1, 1]):
        self.__init_layers_from_dimensions(dims)


    def __call__(self, X):
        for L in self.layers:
            X = L(X)
        return X


   # Train the Perzeptron once given a list of patterns using back 
   # propagation and the Widrow-Hoff rule.
    def train(self, patterns):
        if isinstance(patterns, str):
            patterns = parse_training_file(patterns)

        # First gather all weight changes
        for P in patterns:
            if P.N == self.layers[0].N and P.M == self.layers[-1].M:
                self.__train_pattern(P)
            else:
                raise ValueError("Training input dimension mismatch. \n")

        # Apply weight changes only after all patterns have been applied.
        for layer in self.layers:
            layer.apply_pending_weights()


    # Find the average and maximum error for a list of patterns.    
    def verify(self, patterns):
        if isinstance(patterns, str):
            patterns = parse_training_file(patterns)

        total_error = 0.0
        max_error = 0.0
        for P in patterns:
            total_error += Util.quadratic_error(self(P.X), P.Y)
            if total_error > max_error:
                max_error = total_error

        return total_error / len(patterns), max_error


    # Set learning factor of the ith layer.
    def set_learning_factor(self, learning_factor, i):
        self.layers[i].set_learning_factor(learning_factor)


    # Set transfer function of the ith layer.
    def set_transfer_function(self, transfer_function, i):
        self.layers[i].set_transfer_function(transfer_function)


    # Given a list [N1, N2, N3, ...] of dimensions, create layers N1 -> N2 -> ...
    def __init_layers_from_dimensions(self, dims):
        self.layers = []
        for i in xrange(1, len(dims)):
            self.layers.append(PerzLayer(dims[i-1], dims[i]))


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

        # Compute deltas (delta_h = sum_k delta_k w_hk) as vector.
        W_bar = np.transpose(next_layer.W[:, 1:])
        cur_layer.set_deltas(np.multiply(np.dot(W_bar, delta_bar).flatten(), fprime_net)) 

        # Update delta_weights
        new_delta_weights = cur_layer.learning_factor * \
                            (np.matmul(cur_layer.deltas, prev_out))
        cur_layer.update_pending_weights(new_delta_weights)


    def __repr__(self):
        out = ""
        for layer in self.layers:
            out += str(layer) + "\n"
        return out


# A training pattern consisting of an input vector and a desired output vector.
class Pattern:
    def __init__(self, X, Y):
        if X.ndim == 1 and Y.ndim == 1:
            self.X = X
            self.Y = Y
        
            self.N = X.shape[0]
            self.M = Y.shape[0]
        
        else:
            raise ValueError("Pattern should be constructed from two vectors.")


    def __repr__(self):
        return "Pattern (" + str(self.N) + "):" + "X = " + str(self.X) +  ", Y = " + str(self.Y)


# Parse a file containing training patterns in the following format:
# For each pattern, one line should be provided, containing first the entries
# of the input vector, separated by a single space, then at least 2 spaces,
# followed by the entries of the desired output vector, separated by at keast one space.
# The behaviour of this function is undefined for wrongly formatted input files!
def parse_training_file(filepath):
    file = open(filepath)
    lines = file.readlines()
    patterns = []
    for line in lines:
        if line[0] == "#":
            continue
        spl = re.split("   +", line)
        if len(spl) < 2:
            spl = re.split("\t", line)
        X = np.array([float(x) for x in re.split(" +", spl[0].strip())])
        Y = np.array([float(x) for x in re.split(" +", spl[1].strip())])
        patterns.append(Pattern(X, Y))    
    return patterns

def demo():

    random.seed(12345)

    print "Initializing Perzeptron: "
    q = Perzeptron([4, 8, 2])
    print q
    
    print "Training perzeptron with patterns in ./training.dat 250 times..."

    plotfile = open("learning.curve", 'w')
    for dummy in xrange(250):
        q.train("training.dat")
        error = q.verify("training.dat")[0]
        plotfile.write(str(error) + "\n")
    print "Training done, learning curve written to ./learning.curve."
    print "Verifying perzeptron using patterns in ./test.dat:"
    avg_error, max_error = q.verify("test.dat")

    print "Average quadratic error: " + str(avg_error) 
    print "Maximum quadratic error: " + str(max_error)


if __name__ == "__main__":
    demo()
