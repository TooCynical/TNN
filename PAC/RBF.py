import numpy as np
import re
import random
import Util

def squaredDistance(v, w):
    if len(v) != len(w):
        raise ValueError("Vectors should be of same size!")

    return sum([np.power(a-b,2) for a,b in zip(v, w)])

class RBFNetwork(object):
    
    class Center(object):
        def __init__(self, pos, size):
            self.pos = pos
            self.size = size

        def __call__(self, X):
            return np.exp(-squaredDistance(self.pos, X)/(2 * np.power(self.size, 2)))
    
    # Create an RBF network with N input neurons, K RBF neurons and M output neurons using
    # a given set of patterns. The learning rate used for each weight update is identical.
    # The Gaussian bell function is used as radial function and the usual euclidean norm is
    # used as distance function.
    def __init__(self, N, K, M, patterns, learningRate=0.05):
        self.N = N
        self.K = K
        self.M = M

        self.learningRate = learningRate

        self.__initializePendingWeightChanges()
        self.__initializeWeights()
        self.__initializeCenters(patterns)

    def train(self, patterns):

        # First gather all weight changes
        for P in patterns:
            if P.N == self.N and P.M == self.M:
                self.__trainPattern(P)
            else:
                raise ValueError("Training input dimension mismatch. \n")

        # Apply weight changes only after all patterns have been applied.
        self.__applyWeightChanges()

    def verify(self, patterns):
        errors = [Util.quadratic_error(self(P.X), P.Y) for P in patterns]
        return sum(errors) / len(patterns), max(errors)

    # Returns the result of applying the network to some input
    def __call__(self, X):
        return self.weights.dot(self.__centerResults(X))

    def __centerResults(self, X):
        return np.array([c(X) for c in self.centers])

    def __trainPattern(self, pattern):
        centerResults = self.__centerResults(pattern.X)
        Y = self.weights.dot(centerResults)
        difference = np.array(pattern.Y) - Y
        self.pendingWeights += self.learningRate * np.outer(difference, centerResults)

    def __initializePendingWeightChanges(self):
        self.pendingWeights = np.zeros((self.M, self.K))
        
    def __initializeWeights(self):
        self.weights = np.random.uniform(-0.5, 0.5, (self.M, self.K))

    def __applyWeightChanges(self):
        self.weights += self.pendingWeights
        self.__initializePendingWeightChanges()

    # Input data driven approach to initialize center placements and sizes
    def __initializeCenters(self, patterns):
        # Randomly choose K patterns and use their input data as center placements.
        centerPositions = [p.X for p in random.sample(set(patterns), self.K)]

        # As center sizes use 1/K times the diagonal length of the input space.
        lengths = []
        for i in range(0, self.N):
            dimValues = [p.X[i] for p in patterns]
            lengths.append(max(dimValues) - min(dimValues))
        
        longestDiagonal = np.sqrt(sum([np.power(l,2) for l in lengths]))

        self.centers = [self.Center(p, longestDiagonal/self.K) for p in centerPositions]

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
    
    random.seed(1234)

    patterns = parse_training_file("training.dat")
    testData = parse_training_file("test.dat")

    n = RBFNetwork(4, 6, 2, patterns)

    plotfile = open("learning.curve", 'w')
    for dummy in range(50):
        n.train(patterns)
        error = n.verify(patterns)[0]
        plotfile.write(str(error) + "\n")

    avg_error, max_error = n.verify(testData)

    print("Average quadratic error: " + str(avg_error))
    print("Maximum quadratic error: " + str(max_error))


if __name__ == "__main__":
    demo()
