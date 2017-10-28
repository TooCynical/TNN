import random

# random.seed(1)

def uniform_sampler(x_min, x_max):
	def out():
		return random.uniform(x_min, x_max)
	return out
