import numpy as np


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


# Standard transfer function. You can make more, but have to set the derivative manually!
f = tanh()
tanh_transfer = TransferFunction(f, lambda x: 1 - f(x)**2)

g = logistic()
logistic_transfer = TransferFunction(g, lambda x: (1 - g(x)) * g(x))

h = identity()
identity_transfer = TransferFunction(h, lambda x: 1)