import numpy as np

def simpleRandomInit(arh, seed=None):
	# Seed if provided
	np.random.seed(seed)

	# Random initialization
	ws, bs=[],[]
	for i in range(1,len(arh)):
		bs.append(np.random.randn(arh[i],1))
		ws.append(np.random.randn(arh[i-1], arh[i]).T)
	return ws, bs
