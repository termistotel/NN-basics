import numpy as np

def simpleRandomInit(arh):
	# Random initialization
	ws, bs=[],[]
	for i in range(1,len(arh)):
		bs.append(np.random.randn(arh[i],1))
		ws.append(np.random.randn(arh[i-1], arh[i]).T)
	return ws, bs
