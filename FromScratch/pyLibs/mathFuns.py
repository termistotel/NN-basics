import numpy as np

relu = lambda x: np.maximum(0,x)
step = lambda x: x>0
sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

def logCost(neurons, result):
	M = neurons.shape[1]
	neurons[neurons > 0.9999999] = 0.9999999
	neurons[neurons < 0.0000001] = 0.0000001

	out = -result*np.log(neurons)/M - (1-result)*np.log(1-neurons)/M
	return out