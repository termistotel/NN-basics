import numpy as np
from pyLibs.randomInit import simpleRandomInit
from pyLibs.forwardProp import simpleForwardProp
from pyLibs.gradient import simpleGrad
from pyLibs.learning import batchGradientDescent, miniBatchGradientDescent, gradDescUpdate, momentumGDUpdate, adamGDUpdate

import pyLibs.getData as data
import pyLibs.mathFuns as m

from functools import partial
import random

def callback(X, Y, fp, As, Zs, Ws, bs, funs, cost, iterno):
	if iterno == -1:
		_, As = zip(*fp(X, Ws, bs, funs))
		print(np.sum(cost(Y, As[-1])))
	pass

if __name__ == "__main__":
	# Test on xor
	trainData = data.getIris()
	arh = [trainData[0].shape[0]] + [10, 10, 10] + [trainData[1].shape[0]]

	# Neural net initialization
	seed = random.randint(0, 2**32)
	seed = 1337
	niter = 50
	forwardprop = simpleForwardProp
	gradient = simpleGrad
	funs, grads = [m.relu, m.relu, m.relu, m.sigmoid], [m.step, m.step, m.step, m.dsigmoid]

	cost = m.logCost

	hparameters = {"alpha": 0.1,
				"betav": 0.9,
				"betas": 0.999,
				"epsilon": 0.0001,
				"lambda": 10,
				"batch_size": 512}

	# Ws, bs = simpleRandomInit(arh, 1337)
	# batchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters,
	# 				niter=10*niter, forwardprop=forwardprop, gradient=gradient, numericCompare=False,
	# 				callback=partial(callback, trainData[0], trainData[1], forwardprop),
	# 				updateFun=gradDescUpdate)

	hparameters = {"alpha": 0.008,
				"betav": 0.9,
				"betas": 0.999,
				"epsilon": 0.0001,
				"lambda": 10,
				"batch_size": 512}

	Ws, bs = simpleRandomInit(arh, seed)
	miniBatchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters,
					niter=niter, forwardprop=forwardprop, gradient=gradient,
					callback=partial(callback, trainData[0], trainData[1], forwardprop),
					updateFun=gradDescUpdate )

	hparameters = {"alpha": 0.0057,
				"betav": 0.9,
				"betas": 0.999,
				"epsilon": 0.001,
				"lambda": 10,
				"batch_size": 512}

	Ws, bs = simpleRandomInit(arh, seed)
	miniBatchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters,
					niter=niter, forwardprop=forwardprop, gradient=gradient,
					callback=partial(callback, trainData[0], trainData[1], forwardprop),
					updateFun=momentumGDUpdate )

	hparameters = {"alpha": 0.1048,
				"betav": 0.9,
				"betas": 0.999,
				"epsilon": 0.001,
				"lambda": 10,
				"batch_size": 512}

	Ws, bs = simpleRandomInit(arh, seed)
	miniBatchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters,
					niter=niter, forwardprop=forwardprop, gradient=gradient,
					callback=partial(callback, trainData[0], trainData[1], forwardprop),
					updateFun=adamGDUpdate )
