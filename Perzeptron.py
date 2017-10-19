# Lucas Slot,   lfh.slot@gmail.com  (2984451).
# Julian W.,    jjwarg@gmail.com    (???????)
#
# October 2017
# University of Bonn
#
# Perzepetron.py
# Implementation of a perzeptron without hidden layers.


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
        patterns.append(pattern(X, Y))
    
    return patterns
    

# Returns the 0-1 heaviside function centered at x.
def heaviside(x):
    def out(y):
        return y >= x
    return out

# Implementation of a dual layer perzeptron with BIAS.
# The perzeptron can either be constructed from a weight matrix, or from
# two integers N, M. In the latter case, an (N+1)xM weight matrix will be generated
# at random (entries picked uniformly at random from [-0.5, 0.5]).
# The first column of the weight matrix is reserved for the BIAS.
# Optional input for the constructor are:
#     An initial learning factor (default = 0.1),
#     A transfer function (default is the heaviside function centered at 0).
#
# Callable functions for a perzeptron P are:
#     P(X)                      - evaluate P at X for an input vector X of the right size.
#     P.train(patterns)       - train P by the Widrow-Hoff learning rule using a list of training patterns.
#     P.verify(patterns)      - verify that P's output matches the given patterns.
#     P.set_weights(W)          - manually set P's weights by providing an array or the path
#                                 of a file containing an array. This may change the dimensions of P.
#     P.set_learning_factor(b)  - manually set P's learn factor.
#      
class perzeptron:
    def __init__(self, N=1, M=1, learning_factor=0.1, transfer=heaviside(0)):
        self.__init_weights_from_dimensions(N, M)
        self.__init_transfer(transfer)
        self.__init_learning_factor(learning_factor)


    # Evaluate the perzeptron for given input by applying the transfer function
    # to each entry of W * [1 X]^T.
    def __call__(self, X):
        return self.transfer_vec(np.dot(self.W[:, 1:], X) + self.W[:, 0]).flatten()

        
    # Train the perzeptron x times using the Widrow-Hoff rule by providing a list of patterns
    # or a filepath to a training data file.
    def train(self, patterns, x=1):
        if isinstance(patterns, str):
            patterns = parse_training_file(patterns)
        for dummy in xrange(x):
            self.__train(patterns)


    # Assert that the perzeptron is consistent with a list of patterns.
    def verify(self, patterns):
        if isinstance(patterns, str):
            patterns = parse_training_file(patterns)
        for P in patterns:
            if P.N == self.N and P.M == self.M:
                if not self.__verify_pattern(P):
                    return False
            else:
                raise ValueError("Verification input dimension mismatch. \n")
        return True
                
    
    # Set the weights of the perzeptron either by providing an array or the
    # filepath of a file containing an array.            
    def set_weights(self, new_W):
        if isinstance(new_W, str):
            self.__set_weights_from_array(np.loadtxt(new_W))
        else:
            self.__set_weights_from_array(new_W)

        
    def set_learning_factor(self, new_learning_factor):
        self.learning_factor = new_learning_factor

        
    def __set_weights_from_array(self, new_W):
        new_M = new_W.shape[0]
        if len(new_W.shape) >= 2:
            new_N = new_W.shape[1]
        else:
            new_N = 1
        if self.N + 1 == new_N and self.M == new_M:
            self.W = new_W
        else:
            raise ValueError("New weight dimension mismatch. \n")
 

   # Train the perzeptron once using the Widrow-Hoff rule and a list of patterns.
    def __train(self, patterns):
        for P in patterns:
            if P.N == self.N and P.M == self.M:
                self.__train_pattern(P)
            else:
                raise ValueError("Training input dimension mismatch. \n")    

    
    # Apply the Widrow-Hoff rule to all weights and the BIAS for a single pattern.            
    def __train_pattern(self, P):
        # Compute perzeptron output.
        Y = self(P.X)
        
        # Update weights.
        for n in xrange(self.N):
            for m in xrange(self.M):
                self.W[m][n+1] += self.learning_factor * (P.Y[m] - Y[m]) * P.X[n]
                
        # Update BIAS.
        for m in xrange(self.M):
            self.W[m][0] += self.learning_factor * (P.Y[m] - Y[m]) * 1.0
            
            
     # Verify given pattern is consistent with perzeptron.        
    def __verify_pattern(self, P):
        return (self(P.X) == P.Y).all()


    # Initialize weights as an NxM matrix of values picked uniformly at random
    # from [-0.5, 0.5].
    def __init_weights_from_dimensions(self, N, M):
        if N <= N_MAX and M <= M_MAX:
            self.N = N
            self.M = M
            self.W = np.random.rand(M, N + 1) - 0.5
        else:
            raise ValueError("perzeptron dimensions exceed bounds. \n")


    def __init_learning_factor(self, learning_factor):
        self.learning_factor = learning_factor


    def __init_transfer(self, transfer):                
        self.transfer = transfer
        self.transfer_vec = np.vectorize(self.transfer)

            
    def __repr__(self):
        return "perzeptron (" + str(self.N) + "->" + str(self.M) \
               + "), learning factor: " + str(self.learning_factor) \
               + ",\nWeights: \n" + str(self.W)

               
# A training pattern consisting of an input vector and a desired output vector.
class pattern:
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

           
# Demonstrate this modules functionality.
def demo():
        print "Initializing perzeptron on 3 inputs and 2 outputs:"
        p = perzeptron(3, 2, transfer=heaviside(0))
        print p
        
        print "Training perzeptron 50 times using numpy arrays representing (!XOR, XOR) as patterns:"
        pattern1 = pattern(np.array([0, 0, 0]), np.array([1, 0]))
        pattern2 = pattern(np.array([0, 0, 1]), np.array([0, 1]))
        pattern3 = pattern(np.array([0, 1, 0]), np.array([0, 1]))
        pattern4 = pattern(np.array([0, 1, 1]), np.array([1, 0]))
        patterns = [pattern1, pattern2, pattern3, pattern4]        
        p.train(patterns, 50)
        print p
    
        print "Verifying that perzeptron now implements these patterns (it shouldn't):"
        print p.verify(patterns)
    
        print "Training perzeptron 100 times using patterns in /train_3_2_OR.dat representing (OR, OR):"
        patterns = parse_training_file("train_3_2_OR.dat")
        p.train(patterns, 100)
        print p
        
        print "Verifying that perzeptron now implements the patterns in /train_3_2_OR.dat (it should):"
        print p.verify(patterns)
    
        print "Setting perzeptron weights using a numpy array:"
        p.set_weights(np.array([[-0.2, 0.1, 0.3, 0.3], [0.1, 1, 2, 3]]))
        print p
    
        print "Setting perzeptron weight from /weights.dat:"
        p.set_weights("weights.dat")
        print p


if __name__ == "__main__":
    demo()
