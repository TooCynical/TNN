from __future__ import print_function

import numpy as np

def random_vector(d):
    return np.random.random(d)
    
def length(v):
    return np.sqrt(np.sum(v**2))
    
def dist(v, w):
    return length(v - w)    
    
def angle(v):
    d = len(v)
    return np.arccos(np.sum(v) / (length(v) * np.sqrt(d)))


P = 100
d = 10


f1 = open("length.data")
f2 = open("angle.data")
f3 = open("dist.data")

V = []
for p in range(P):
    V.append(random_vector(d))
    
for v in V:
    print(length(v), file=f1)

for v in V:
    print(angle(v), file=f2)

    
for v, w in zip(V[:-1], V[1:]):
    print(dist(v, w), file=f3)

