import numpy as np
from pyLibs.randomInit import simpleRandomInit
from pyLibs.forwardProp import simpleForwardProp
from pyLibs.gradient import simpleGrad
from pyLibs.learning import batchGradientDescent, miniBatchGradientDescent, gradDescUpdate, momentumGDUpdate

import pyLibs.getData as data
import pyLibs.mathFuns as m

from functools import partial

def callback(X, Y, fp, As, Zs, Ws, bs, funs, cost, iterno):
	if iterno%100 == 0:
		_, As = zip(*fp(X, Ws, bs, funs))
		print(np.sum(cost(Y, As[-1])))
	pass

if __name__ == "__main__":
	# Test on xor
	trainData = data.getIris()

	arh = [trainData[0].shape[0]] + [20] + [trainData[1].shape[0]]

	# Neural net initialization
	niter = 50
	forwardprop = simpleForwardProp
	gradient = simpleGrad
	Ws, bs = simpleRandomInit(arh)
	funs, grads = [m.relu, m.sigmoid], [m.step, m.dsigmoid]
	hparameters = {"alpha": 0.01,
				"betav": 0.9,
				"betas": 0.999,
				"epsilon": 0.0001,
				"lambda": 10,
				"batch_size": 512}
	cost = m.logCost

	Ws1, bs1, Ws2, bs2 = [], [], [], []
	for W, b in zip(Ws, bs):
		Ws1.append(W.copy())
		bs1.append(b.copy())
		Ws2.append(W.copy())
		bs2.append(b.copy())

	batchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters, 
					niter=niter*10, forwardprop=forwardprop, gradient=gradient, numericCompare=False,
					callback=partial(callback, trainData[0], trainData[1], forwardprop),
					updateFun=gradDescUpdate)

	# input()

	miniBatchGradientDescent(trainData, Ws1, bs1, funs, grads, cost, hparameters,
					niter=niter, forwardprop=forwardprop, gradient=gradient,
					callback=partial(callback, trainData[0], trainData[1], forwardprop),
					updateFun=gradDescUpdate )


	hparameters = {"alpha": 0.4,
				"betav": 0.9,
				"betas": 0.999,
				"epsilon": 0.0001,
				"lambda": 10,
				"batch_size": 512}

	miniBatchGradientDescent(trainData, Ws2, bs2, funs, grads, cost, hparameters,
					niter=niter, forwardprop=forwardprop, gradient=gradient,
					callback=partial(callback, trainData[0], trainData[1], forwardprop),
					updateFun=momentumGDUpdate )
