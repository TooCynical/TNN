# Lucas S.   	lfh.slot@gmail.com
# Julian W.    	jjwarg@gmail.com  
#
# University of Bonn
# November 2017
#
# Util.py
# Utility functions for the implementation of an MSOM.


import numpy as np
import random


def exp_decay(y_0, y_final, x_final):
	def out(x):
		return np.exp(-3 * (x / x_final)) * (y_0 - y_final) + y_final
	return out

def uniform_sampler(x_min, x_max):
	def out():
		return random.uniform(x_min, x_max)
	return out


# Return half of the quadratic distance X - Y.
def quadratic_error(X, Y):
    return np.sum((X - Y)**2)


class TransferFunction:
	def __init__(self, f, f_prime):
		self.f = f
		self.f_prime = f_prime


def heaviside(shift=0):
    def out(y):
        return y >= shift
    return out


def tanh(xshift=0.0, yshift=0.0, xscale=1.0, yscale=1.0):
	def out(x):
		return np.tanh(xshift + xscale * x) * yscale + yshift
	return out


def logistic(xshift=0.0, yshift=0.0, xscale=1.0, yscale=1.0):
	def out(x):
		return 1.0 / (np.exp(-(xshift + xscale * x)) + 1) * yscale + yshift
	return out


def identity(xshift=0.0, yshift=0.0, xscale=1.0, yscale=1.0):
	def out(x):
		return (xshift + xscale * x) * yscale + yshift
	return out


# Standard transfer functions. You can make more, but have to set the derivative manually!
f = tanh()
tanh_transfer = TransferFunction(f, lambda x: 1 - f(x)**2)

g = logistic()
logistic_transfer = TransferFunction(g, lambda x: (1 - g(x)) * g(x))

h = identity()
identity_transfer = TransferFunction(h, lambda x: 1)