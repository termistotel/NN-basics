import numpy as np

def simpleForwardProp(X, Ws, bs, funs):
	if (Ws == []) or (bs ==[]) or (funs == []):
		return []
	Z = Ws[0].dot(X) + bs[0]
	A = funs[0](Z)
	return [(Z, A)] + simpleForwardProp(A, Ws[1:], bs[1:], funs[1:])
