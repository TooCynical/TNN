# -*- coding: utf-8 -*-
import numpy as np

N_MAX = 101
M_MAX = 30
T_MAX = 201

# Parse a file containing training patterns in the following format:
#   - The first line should contain three integers N, M, t, where N, M
#     are the dimensions of the training input / output, respectively, and
#     t is the total amount of training patterns.
#   - For each pattern, two lines should be provided, the first one containing
#     the entries of the training input, and the second one the entries of the
#     training output, seperated by spaces.
#
# The behaviour of this function is undefined for wrongly formatted input files!
def parse_training_file(filepath):
    file = open(filepath)
    lines = file.readlines()
    first_line_ints = [int(x) for x in lines[0].split(" ")]
    N = first_line_ints[0]
    M = first_line_ints[1]
    t = first_line_ints[2]
    
    if t > T_MAX:
        raise ValueError("Amount of patterns in data file exceeds bounds. \n")

    patterns = []
    for i in xrange(t):
        X = np.fromstring(lines[1 + 2*i], sep=" ")
        Y = np.fromstring(lines[2 + 2*i], sep=" ")
        if X.shape[0] != N or Y.shape[0] != M:
            raise ValueError("Training file dimension mismatch. \n")
        patterns.append(Pattern(X, Y))
    
    return patterns
    

# Returns the 0-1 heaviside function centered at x.
def heaviside(x):
    def out(y):
        return y >= x
    return out

# Implementation of a dual layer Perzeptron with BIAS.
# The Perzeptron can either be constructed from a weight matrix, or from
# two integers N, M. In the latter case, an NxM weight matrix will be generated
# at random (entries picked uniformly at random from [-0.5, 0.5]).
# Optional input for the constructor are:
#     An initial BIAS (default = 0),
#     An initial learning factor (default = 0.1),
#     A transfer function (default is the heaviside function centered at 0).
#
# Callable functions for a Perzeptron P are:
#     P(X)                  - evaluate P at X for an input vector X of the right size.
#     P.train([patterns])   - train P by the Widrow-Hoff learning rule using a list of training patterns.
#     P.verify([patterns])  - verify that P's output matches the given patterns.
#     P.set_weights(W)      - manually set P's weights by providing an array or the path
#                             of a file containing an array. This may change the dimensions of P.
#     P.set_BIAS(b)         - manually set P's BIAS.
#     P.set_learn_factor(b) - manually set P's learn factor.
#      
class Perzeptron:
    def __init__(self, N=1, M=1, W=None, BIAS=0.0, learn_factor=0.1, transfer=heaviside(0)):
        if W is None:
            self.__init_weights_from_dimensions(N, M)
        else:
            self.__init_weights_from_array(W)

        self.__init_transfer(transfer)
        self.__init_BIAS(BIAS)
        self.__init_learning_factor(learn_factor)


    # Evaluate the perzeptron for given input.
    def __call__(self, X):
       return self.transfer_vec(np.dot(self.W, X) + self.BIAS)

        
    # Train the perzeptron x times using the Widrow-Hoff rule by providing a list of patterns.
    def train(self, patterns, x=1):
        for dummy in xrange(x):
            self.__train(patterns)

    
    def verify(self, patterns):
        for P in patterns:
            if P.N == self.N and P.M == self.M:
                if not self.__verify_pattern(P):
                    return False
            else:
                raise ValueError("Verification input dimension mismatch. \n")
        return True
                
    
    # Set the weights of the Perzeptron either by providing an array or the
    # filepath of a file containing an array.            
    def set_weights(self, new_W):
        if isinstance(new_W, str):
            self.__set_weights_from_array(np.loadtxt(new_W))
        else:
            self.__set_weights_from_array(new_W)


    def set_BIAS(self, new_BIAS):
        self.BIAS = new_BIAS

        
    def set_learning_factor(self, new_learn_factor):
        self.learn_factor = new_learn_factor

        
    def __set_weights_from_array(self, new_W):
        new_N = new_W.shape[0]
        if len(new_W.shape) >= 2:
            new_M = new_W.shape[1]
        else:
            new_M = 1
        if self.N == new_N and self.M == new_M:
            self.W = new_W
        else:
            raise ValueError("New weight dimension mismatch. \n")
    # Train the Perzeptron once using the Widrow-Hoff rule and a list of patterns.
    def __train(self, patterns):
        for P in patterns:
            if P.N == self.N and P.M == self.M:
                self.__train_pattern(P)
            else:
                raise ValueError("Training input dimension mismatch. \n")    
    
    # Apply the Widrow-Hoff rule to all weights and the BIAS for a single pattern.            
    def __train_pattern(self, P):
        # Compute Perzeptron output.
        Y = self(P.X)
        
        # Update weights.
        for n in xrange(self.N):
            for m in xrange(self.M):
                self.W[m][n] += self.learn_factor * (P.Y[m] - Y[m]) * P.X[n]
                
        # Update BIAS.
        for m in xrange(self.M):
            self.BIAS += self.learn_factor * (P.Y[m] - Y[m]) * 1.0
            
            
     # Verify given pattern is consistent with Perzeptron.        
    def __verify_pattern(self, P):
        return (self(P.X) == P.Y).all()


    # Initialize weights from a provided array and set N, M accordingly.
    def __init_weights_from_array(self, W):
        self.W = W
        self.N = W.shape[0]
        self.M = W.shape[1]


    # Initialize weights as an NxM matrix of values picked uniformly at random
    # from [-0.5, 0.5].
    def __init_weights_from_dimensions(self, N, M):
        if N <= N_MAX and M <= M_MAX:
            self.N = N
            self.M = M
            self.W = np.random.rand(M, N) - 0.5
        else:
            raise ValueError("Perzeptron dimensions exceed bounds. \n")

            
    def __init_BIAS(self, BIAS):
        self.BIAS = BIAS

            
    def __init_learning_factor(self, learn_factor):
        self.learn_factor = learn_factor


    def __init_transfer(self, transfer):                
        self.transfer = transfer
        self.transfer_vec = np.vectorize(self.transfer)

            
    def __repr__(self):
        return "Perzeptron (" + str(self.N) + "->" + str(self.M) \
               + "), learning factor: " + str(self.learn_factor) \
               + ",\nWeights: \n" + str(self.W) + "\n" \
               + "BIAS: " + str(self.BIAS)

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
            raise ValueError("Pattern should be constructed from two vectors. \n")
    
    
    # Return whether the input and output of this pattern are binary.
    def is_binary(self):
        return ((self.X == 0) | (self.X == 1)).all() and ((self.Y == 0) | (self.Y == 1)).all()


if __name__ == "__main__":
    p = Perzeptron(3, 1, transfer=heaviside(0))
    
#    X1 = np.array([0, 0])
#    Y1 = np.array([0])
#    X2 = np.array([0, 1])
#    Y2 = np.array([0])
#    X3 = np.array([1, 0])
#    Y3 = np.array([0])
#    X4 = np.array([1, 1])
#    Y4 = np.array([1])
#    
#    P1 = Pattern(X1, Y1)
#    P2 = Pattern(X2, Y2)
#    P3 = Pattern(X3, Y3)
#    P4 = Pattern(X4, Y4)
    
#    for x in xrange(100):
#        p.train([P1, P2, P3, P4])
    
    patterns = parse_training_file("train_3_1_OR.dat")
    p.train(patterns, 100)
    print p.verify(patterns)
    print p

    p.set_weights(np.array([0.1, 0.3, 0.3]))
    print p

    p.set_weights("weights.dat")
    print p
