# -*- coding: utf-8 -*-
import random
import numpy as np
import numbers

N_MAX = 101
M_MAX = 30

# Representation of a dual layer Perzeptron with binary output.
class Perzeptron:
    # Construct a Perzeptron either by providing a weight matrix W explicitely,
    # or by providing two integers N, M, in which case an NxM weight matrix 
    # will be generated at random (entries taken uniformly from [-0,5, 0.5]).
    # If neither are provided a 1x1 perzeptron with random weight will be initialized. 
    # Initial BIAS and learn_factor may also be provided explicitely, 
    # or left at default (0, 0.1 resp.)
    def __init__(self, N=1, M=1, W=None, BIAS=None, learn_factor=None):
        if W is None:
            self.__init_weights_from_dimensions(N, M)
        else:
            self.__init_weights_from_array(W)
            
        self.__init_BIAS(BIAS)
        self.__init_learning_factor(learn_factor)

    # Set the weights of the Perzeptron.
    def set_weights(self, new_W):
        if isinstance(new_W, np.ndarray):
            if new_W.shape[0] == self.N and new_W.shape[1] == self.M:
                self.W = new_W
            else:
                raise ValueError("Invalid Perzeptron weight dimensions. \n")
        else:
            raise ValueError("Invalid Perzeptron weights. \n")
    
    # Evaluate the perzeptron for given input.
    def __call__(self, X):
        try:
            output = (np.dot(self.W, X) >= -self.BIAS)
        except ValueError:
            raise ValueError("Invalid Perzeptron input. \n")
        return output
        
    # Train the perzeptron using the Widrow-Hoff rule by providing a list of Patterns.
    def train(self, patterns):
        for P in patterns:
            if isinstance(P, Pattern):
                if P.N == self.N and P.M == self.M:
                    self.__train_pattern(P)
                else:
                    raise ValueError("Training input dimension mismatch. \n")
            else:
                raise TypeError("Training input must be a list of Patterns. \n")
    
    # Apply the Widrow-Hoff rule to all weights and the BIAS for a single Pattern.            
    def __train_pattern(self, P):
        # Compute Perzeptron output.
        Y = self(P.X)
        
        # Update weights.
        for n in xrange(self.N):
            for m in xrange(self.M):
                self.W[m][n] += self.learn_factor * (P.Y[m] - Y[m]) * P.X[n]
                
        # Update BIAS.
        for m in xrange(self.M):
            self.BIAS += self.learn_factor * (P.Y[m] - Y[m])

    # Initialize weights from a provided array and set N, M accordingly.
    def __init_weights_from_array(self, W):
        if isinstance(W, np.ndarray):
            self.W = W
            self.N = W.shape[0]
            self.M = W.shape[1]
        else:
            raise ValueError("Invalid initial Perzeptron weights. \n")
        
    # Initialize weights as an NxM matrix of values picked uniformly at random
    # from [-0.5, 0.5].
    def __init_weights_from_dimensions(self, N, M):
        if isinstance(N, int) and isinstance(M, int) and N > 0 and M > 0:
            if N <= N_MAX and M <= M_MAX:
                self.N = N
                self.M = M
                self.W = np.random.rand(M, N) - 0.5
            else:
                raise ValueError("Perzeptron dimensions exceed bounds. \n")
        else:
            raise TypeError("Perzeptron dimensions should be positive integers")
            
    def __init_BIAS(self, BIAS):
        if BIAS is None:
            self.BIAS = 0.
        elif isinstance(BIAS, numbers.Number):
            self.BIAS = BIAS
        else:
            raise TypeError("BIAS should be a real number. \n")
            
    def __init_learning_factor(self, learn_factor):
        if learn_factor is None:
            self.learn_factor = 0.1
        elif isinstance(learn_factor, numbers.Number):
            self.learn_factor = learn_factor
        else:
            raise TypeError("Learning factor should be a real number. \n")
            
    def __repr__(self):
        return "Perzeptron: " + str(self.W.shape) + ", BIAS: " + str(self.BIAS) \
               + ", Learning factor: " + str(self.learn_factor) \
               + ", Weights: \n" + str(self.W)

# A training pattern consisting of an input vector and a desired output vector.
class Pattern:
    # Construct a pattern from input and desired output.
    def __init__(self, X, Y):
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            if X.ndim == 1 and Y.ndim == 1:
                self.X = X
                self.Y = Y
                self.N = X.shape[0]
                self.M = Y.shape[0]
            else:
                raise ValueError("Pattern should be constructed from two vectors. \n")
        else:
            raise ValueError("Pattern should be constructed from two vectors. \n")
    
    # Return whether the input and output of this pattern are binary.
    def is_binary(self):
        return ((self.X == 0) | (self.X == 1)).all() and ((self.Y == 0) | (self.Y == 1)).all()

p = Perzeptron(2, 2)

X1 = np.array([0, 0])
Y1 = np.array([0, 0])
X2 = np.array([0, 1])
Y2 = np.array([0, 0])
X3 = np.array([1, 0])
Y3 = np.array([0, 0])
X4 = np.array([1, 1])
Y4 = np.array([1 ,1])

P1 = Pattern(X1, Y1)
P2 = Pattern(X2, Y2)
P3 = Pattern(X3, Y3)
P4 = Pattern(X4, Y4)

print P4.is_binary()

print p

for x in xrange(100):
    p.train([P1, P2, P3, P4])
print p
