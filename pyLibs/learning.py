import numpy as np
from multiprocessing.dummy import Pool 

from pyLibs.forwardProp import simpleForwardProp
from pyLibs.gradient import simpleGrad, numericGrad

from functools import wraps, partial
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

def gradDescUpdate(cache):
	# Extract variables
	X, Y = cache["X"], cache["Y"]
	Ws, bs = cache["Ws"], cache["bs"]
	funs, grads, cost = cache["funs"], cache["grads"], cache["cost"]
	hparameters, forwardprop, gradient = cache["hparameters"], cache["forwardprop"], cache["gradient"]
	callback, numericCompare = cache["callback"], cache["numericCompare"]
	iterno = cache["iterno"]

	# Get Hyperparameters
	lrate = hparameters["alpha"]
	M = Y.shape[1]

	if X.shape[1] < X.shape[0]:
		return

	# Forward propagation
	Zs, As = zip(*forwardprop(X, Ws, bs, funs))
	Zs, As = map(list, (Zs, As))

	# Starting iteration:
	# D = (dj/dZ)(Z) * (dAL/dZ)(Z)
	#
	# For a good choice of cost function and final layer activation function:
	# D = A - Y
	A = As.pop()
	Z = Zs.pop()
	grad = grads[-1]
	D = (A - Y)/(2*M)

	# Calculate gradients
	gradsW, gradsb = zip(*gradient([X]+As, [None] + Zs, Ws, bs, grads[:-1], D))
	gradsW, gradsb = map(lambda x: list(x), (gradsW, gradsb))

	# Calculate numeric gradients for debug
	if numericCompare:
		dWs1, dbs1 = numericGrad(X, Y, Ws, bs, funs, cost, forwardprop, epsilon = 0.0001)
	# Print numeric gradient for debuging purpose
		for dW1, dW in zip(dWs1, gradsW):
			print("numeric gradient difference W: ")
			diff = dW1-dW
			print(diff[diff>0.000001])
		for db1, db in zip(dbs1, gradsb):
			print("numeric gradient difference b: ")
			diff = db1-db
			print(diff[diff>0.000001])

	for W, b, dW, db in zip(Ws, bs, gradsW, gradsb):
		W -= lrate * dW
		b -= lrate * db

	callback(As+[A], Zs+[Z], Ws, bs, funs, cost, iterno)
	return cache

def momentumGDUpdate(cache):
	# Extract variables
	X, Y = cache["X"], cache["Y"]
	Ws, bs = cache["Ws"], cache["bs"]
	funs, grads, cost = cache["funs"], cache["grads"], cache["cost"]
	hparameters, forwardprop, gradient = cache["hparameters"], cache["forwardprop"], cache["gradient"]
	callback, numericCompare = cache["callback"], cache["numericCompare"]
	iterno = cache["iterno"]

	# Get Hyperparameters
	lrate = hparameters["alpha"]
	beta = hparameters["betav"]
	M = Y.shape[1]

	if X.shape[1] < X.shape[0]:
		return

	# Forward propagation
	Zs, As = zip(*forwardprop(X, Ws, bs, funs))
	Zs, As = map(list, (Zs, As))

	# Starting iteration:o
	# D = (dj/dZ)(Z) * (dAL/dZ)(Z)
	#
	# For a good choice of cost function and final layer activation function:
	# D = A - Y
	A = As.pop()
	Z = Zs.pop()
	grad = grads[-1]
	D = (A - Y)/(2*M)

	# Calculate gradients
	gradsW, gradsb = zip(*gradient([X]+As, [None] + Zs, Ws, bs, grads[:-1], D))
	gradsW, gradsb = map(lambda x: list(x), (gradsW, gradsb))

	# add vdWs and vdbs to cache if they dont exist
	if "vdWs" in cache:
		vdWs, vdbs = cache["vdWs"], cache["vdbs"]
	else:
		vdWs, vdbs = [], []
		for W, b in zip(Ws, bs):
			vdWs.append(np.zeros(shape=W.shape))
			vdbs.append(np.zeros(shape=b.shape))

	# update weights and bias'
	for W, b, dW, db, vdW, vdb in zip(Ws, bs, gradsW, gradsb, vdWs, vdbs):
		# print(vdW)
		# print(dW)
		vdW *= beta
		vdW += (1-beta)*dW

		vdb *= beta
		vdb += (1-beta)*db
		# if iterno==4:
		# 	print(vdW)

		# Compensation for vdW and vdb beeing small in early iterations
		vdW_corr = vdW/(1-beta**(iterno+1))
		vdb_corr = vdb/(1-beta**(iterno+1))
		# if iterno==4:
		# 	print(vdW)
		# 	print(1-beta**(iterno+1), 1/(1-beta**(2*iterno+1)))
		# input()

		# Update weights and bias' with vdW and vdb
		W -= lrate * vdW_corr
		b -= lrate * vdb_corr

	cache["vdWs"] = vdWs
	cache["vdbs"] = vdbs

	callback(As+[A], Zs+[Z], Ws, bs, funs, cost, iterno)

	return cache

def adamGDUpdate(cache):
	# Extract variables
	X, Y = cache["X"], cache["Y"]
	Ws, bs = cache["Ws"], cache["bs"]
	funs, grads, cost = cache["funs"], cache["grads"], cache["cost"]
	hparameters, forwardprop, gradient = cache["hparameters"], cache["forwardprop"], cache["gradient"]
	callback, numericCompare = cache["callback"], cache["numericCompare"]
	iterno = cache["iterno"]

	# Get Hyperparameters
	lrate = hparameters["alpha"]
	betav = hparameters["betav"]
	betas = hparameters["betas"]
	epsilon = hparameters["epsilon"]

	# Training data size
	M = Y.shape[1]

	if X.shape[1] < X.shape[0]:
		return

	# Forward propagation
	Zs, As = zip(*forwardprop(X, Ws, bs, funs))
	Zs, As = map(list, (Zs, As))

	# Starting iteration:o
	# D = (dj/dZ)(Z) * (dAL/dZ)(Z)
	#
	# For a good choice of cost function and final layer activation function:
	# D = A - Y
	A = As.pop()
	Z = Zs.pop()
	grad = grads[-1]
	D = (A - Y)/(2*M)

	# Calculate gradients
	gradsW, gradsb = zip(*gradient([X]+As, [None] + Zs, Ws, bs, grads[:-1], D))
	gradsW, gradsb = map(lambda x: list(x), (gradsW, gradsb))

	# add momentum terms vdWs and vdbs to cache if they dont exist
	if "vdWs" in cache:
		vdWs, vdbs = cache["vdWs"], cache["vdbs"]
	else:
		vdWs, vdbs = [], []
		for W, b in zip(Ws, bs):
			vdWs.append(np.zeros(shape=W.shape))
			vdbs.append(np.zeros(shape=b.shape))

	# add RMSprop sdWs and sdbs to cache if they dont exist
	if "sdWs" in cache:
		sdWs, sdbs = cache["sdWs"], cache["sdbs"]
	else:
		sdWs, sdbs = [], []
		for W, b in zip(Ws, bs):
			sdWs.append(np.zeros(shape=W.shape))
			sdbs.append(np.zeros(shape=b.shape))

	# update weights and bias'
	for i in range(len(Ws)):
		dW, db = gradsW[i], gradsb[i]
		# Momentum updates
		vdWs[i] = vdWs[i]*betav + (1-betav)*dW
		vdbs[i] = vdbs[i]*betav + (1-betav)*db

		# Compensation for vdW and vdb beeing small in early iterations
		vdWs_corr = vdWs[i] / (1-betav**(iterno+1))
		vdbs_corr = vdbs[i] / (1-betav**(iterno+1))

		# RMSprop upates
		sdWs[i] = sdWs[i]*betas + (1-betas)*np.square(dW)
		sdbs[i] = sdbs[i]*betas + (1-betas)*np.square(db)

		# Compensation for small terms
		sdWs_corr = sdWs[i] / (1-(betas**(iterno+1)))
		sdbs_corr = sdbs[i] / (1-(betas**(iterno+1)))

		# Update weights and bias' with momentum and RMSprop
		Ws[i] = Ws[i] - lrate * vdWs_corr/(np.sqrt(sdWs_corr)+epsilon)
		bs[i] = bs[i] - lrate * vdbs_corr/(np.sqrt(sdbs_corr)+epsilon)

	cache["vdWs"] = vdWs
	cache["vdbs"] = vdbs
	cache["sdWs"] = sdWs
	cache["sdbs"] = sdbs

	callback(As+[A], Zs+[Z], Ws, bs, funs, cost, iterno)

	return cache

@timing
def batchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters, niter = 100, forwardprop=simpleForwardProp, gradient=simpleGrad, callback=lambda *x: x, numericCompare=False, updateFun=gradDescUpdate):
	""" batch gradient descent training

		niter is the number of iterations
		trainData is a touple of X-s and Y-s
		Ws is the list of weights
		bs is the list of bias'
		funs is the list of activation functions for each layer
		grads is the list of derivatives of activation funtions for each layer
		cost is the cost function
		hparameters is a dict of hiperparameters

		forwardprop is a function for calculating forward propagation
		gradient is a function for calculating cost function gradient with respect to weights and bias'

		callback function is a a function that is called every epoch
			it takes argumetns As, Zs, Ws and bs"""

	# Iteration termination
	if niter <= 0:
		return Ws, bs

	# Extracting training data
	X, Y = trainData

	cache= {"X": X, "Y": Y, "Ws": Ws, "bs": bs,
		"funs": funs, "grads": grads, "cost": cost,
		"hparameters": hparameters, "forwardprop": forwardprop, "gradient": gradient,
		"callback": callback, "numericCompare": numericCompare}

	for i in range(niter):
		cache["iterno"] = i
		updateFun(cache)

	callback([], [], Ws, bs, funs, cost, -1)
	return Ws, bs

@timing
def miniBatchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters, niter = 10, forwardprop = simpleForwardProp, gradient = simpleGrad, callback=lambda *x: x, numericCompare=False, updateFun=gradDescUpdate):
	batchSize = hparameters["batch_size"]

	# Iteration termination
	if niter <= 0:
		return Ws, bs

	X, Y = trainData
	M = Y.shape[1]

	cache = {"Ws": Ws, "bs": bs,
		"funs": funs, "grads": grads, "cost": cost,
		"hparameters": hparameters, "forwardprop": forwardprop, "gradient": gradient,
		"callback": callback, "numericCompare": numericCompare}

	def takeBatch(i, iterno, cache):
		X1, Y1 = X[:, i*batchSize:(i+1)*batchSize], Y[:, i*batchSize:(i+1)*batchSize]
		cache["X"] = X1
		cache["Y"] = Y1
		cache["iterno"] = iterno
		cache = updateFun(cache)
		return cache

	for i in range(niter):
		for j in range(int(M/batchSize) + 1):
			cache = takeBatch(j, i*(int(M/batchSize)+1) + j, cache)

	callback([], [], Ws, bs, funs, cost, -1)
	return Ws, bs

@timing
def parallelMiniBatchGradientDescent(trainData, Ws, bs, funs, grads, cost, hparameters, niter = 10, forwardprop = simpleForwardProp, gradient = simpleGrad, callback=lambda *x: x, numericCompare=False, updateFun=gradDescUpdate):
	batchSize = hparameters["batch_size"]

	# Iteration termination
	if niter <= 0:
		return Ws, bs

	X, Y = trainData
	M = Y.shape[1]

	def takeBatch(i):
		X1, Y1 = X[:, i*batchSize:(i+1)*batchSize], Y[:, i*batchSize:(i+1)*batchSize]
		cache= {"X": X1, "Y": Y1, "Ws": Ws, "bs": bs,
			"funs": funs, "grads": grads, "cost": cost,
			"hparameters": hparameters, "forwardprop": forwardprop, "gradient": gradient,
			"callback": callback, "numericCompare": numericCompare, "iterno": i}
		updateFun(cache)

	for i in range(niter):
		pool = Pool()
		# print("Starting epoch " + str(i))
		pool.map(takeBatch, range(int(M/batchSize) + 1))
		pool.close()
		pool.join()

	callback([], [], Ws, bs, funs, cost, -1)
	return Ws, bs
