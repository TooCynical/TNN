# Lucas S.      lfh.slot@gmail.com
# Julian W.     jjwarg@gmail.com  
#
# University of Bonn
# December 2017
#
# ROLF.py
# Implementation of a ROLF network.


import numpy as np
import itertools

import Util

class ROLFNet:

    MAX_NEURONS = 10000
    MAX_INPUT_DIM = 10
    SIGMA_METHOD = "MEAN"

    def __init__(self, N, sigma_init=1.0):
        assert(N < ROLFNet.MAX_INPUT_DIM)
        self.N = N
        self.sigma_init = sigma_init
        self.neurons = []


    # Find the closest neuron in the net that perceives X. If it exists,
    # teach it using X. If it does not exist make a new neuron with center 
    # at X.
    def learn(self, X):
        winning_neuron = None
        best_dist = -1.0
        for neuron in self.neurons:
            perceived, dist = neuron(X)
            if perceived and (dist < best_dist or winning_neuron == None):
                winning_neuron = neuron
                best_dist = dist

        if winning_neuron:
            winning_neuron.learn(X)
        else:
            if len(self.neurons) < ROLFNet.MAX_NEURONS:
                self.__create_new_neuron(X)


    def __create_new_neuron(self, X):
        print "Created new ROLFNeuron..."
        new_neuron = ROLFNeuron(self.N, X, sigma=self.__sigma())
        self.neurons.append(new_neuron)


    # Return a sigma value to be used for a newly created neuron based
    # on some criterion.
    def __sigma(self):
        if ROLFNet.SIGMA_METHOD == "MEAN":
            return self.__mean_sigma()
        if ROLFNet.SIGMA_METHOD == "MIN":
            return self.__min_sigma()
        if ROLFNet.SIGMA_METHOD == "MAX":
            return self.__max_sigma()
        if ROLFNet.SIGMA_METHOD == "INIT":
            return self.sigma_init
        return self.sigma_init


    # Return the minimum value of sigma over all neurons in the net.
    def __min_sigma(self):
        if not self.neurons:
            return self.sigma_init
        else:
            return min(self.__sigmas())


    # Return the minimum value of sigma over all neurons in the net.
    def __max_sigma(self):
        if not self.neurons:
            return self.sigma_init
        else:
            return max(self.__sigmas())


    def __mean_sigma(self):
        if not self.neurons:
            return self.sigma_init
        else:
            return sum(self.__sigmas()) / len(self.neurons)


    # Return a list consisting of the sigma value of each neuron in the net.
    def __sigmas(self):
        return [neuron.sigma for neuron in self.neurons]


    def dump_center_coords_and_sizes(self, filename):
        f = open(filename, 'w')
        for neuron in self.neurons:
            for x in neuron.center.pos:
                f.write(str(x) + " ")
            f.write("   " + str(neuron.sigma))
            f.write("\n")


    def __repr__(self):
        out = "ROLF network (N=" + str(self.N) + ", sigma_init=" + str(self.sigma_init) + ") " +\
        "Having " + str(len(self.neurons)) + " neurons: \n"
        for neuron in self.neurons:
            out += str(neuron) + "\n"
        return out


# A ROLF neuon, consisting of a center and values for
# sigma, rho, eta_sigma and eta_c. Capable of learning
# given an input vector X of the right size.
class ROLFNeuron:
    # A Center. Consists of a dimension N and a vector X in R^N.
    # Capable of returning the euclidean distance from itself to 
    # a given input, and moving itself towards a given input.
    class Center:
        def __init__(self, N, pos):
            self.pos = pos
            self.N = N


        # Return the euclidean distance between this centre and X.
        def __call__(self, X):
            return Util.eucl_dist(self.pos, X)

       
        # Move centre towards X, covering a delta fraction of the distance.
        def update(self, X, delta):
            self.pos += delta * (X - self.pos)
    

        def __repr__(self):
            return str(self.N) + '-center @ ' + str(self.pos)


    def __init__(self, N, pos, rho=2.0, sigma=1.0, eta_sigma=0.05, eta_c=0.05):
        self.N = N
        self.rho = rho
        self.sigma = sigma

        self.eta_sigma = eta_sigma
        self.eta_c = eta_c

        self.center = ROLFNeuron.Center(self.N, pos)


    # Return a pair (b, y) consisting of a boolean b indicating whether
    # X is perceived by this ROLFNeuron, and a real y indicating the 
    # quadratic distance between the center of this ROLFNeuron and X.
    def __call__(self, X):
        distance = self.center(X)
        return self.perceives(distance), distance


    # Return whether given distance is within perceptive range.
    def perceives(self, distance):
        return distance < self.rho * self.sigma


    # Update center and sigma based on X. This only makes sense
    # if X is pereived by this neuron.
    def learn(self, X):
        perceived, distance = self(X)
        assert(perceived)

        # Update center.
        self.center.update(X, self.eta_c)

        # Update sigma.
        self.sigma += self.eta_sigma * (distance - self.sigma)


    def __repr__(self):
        return "ROLFNeuron (N=" + str(self.N) + ", sigma=" + str(self.sigma) + ", rho=" + str(self.rho) + ")"\
        " with center: \n" + str(self.center)


def demo():
    N = ROLFNet(4)
    for X in Util.parse_training_file("test.data"):
        print "Learning " + str(np.array(X)) + "..."
        N.learn(np.array(X))

    print N
    N.dump_center_coords_and_sizes("test.out")
    print "center locations dumped to test.out"


if __name__=="__main__":
    demo()