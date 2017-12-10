# Lucas S.   	lfh.slot@gmail.com
# Julian W.    	jjwarg@gmail.com  
#
# University of Bonn
# November 2017
#
# Util.py
# Utility functions for implementations of various NNs.

import numpy as np
import random
import re

# Return euclidean distance ||X - Y||.
def eucl_dist(X, Y):
    return np.sqrt(np.sum((X - Y)**2))

# Parse a file containing training patterns in the following format:
# For each pattern, one line should be provided, containing first the entries
# of the input vector, separated by a single space, then at least 2 spaces,
# followed by the entries of the desired output vector, separated by at keast one space.
# The behaviour of this function is undefined for wrongly formatted input files!
def parse_training_file(filepath):
    file = open(filepath)
    lines = file.readlines()
    codes = []
    for line in lines:
        if line[0] == "#":
            continue
        X = re.split(" ", line)
        X = [float(x) for x in X]
        codes.append(X)    
    return codes

