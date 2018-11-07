import numpy as np

def simpleGrad(As, Zs, Ws, bs, grads, D, **kwargs):
	if (Ws == []) or (bs ==[]):
		return []

	Wl, bl = Ws[-1], bs[-1]
	Zl, Al = Zs[-1], As[-1]

	dW = D.dot(Al.T)
	db = np.sum(D, axis=1, keepdims=True)

	if Zl is None:
		return [(dW, db)]
	
	grad = grads[-1]
	D = Wl.T.dot(D) * grad(Zl)

	return simpleGrad(As[:-1], Zs[:-1], Ws[:-1], bs[:-1], grads[:-1], D) + [(dW, db)]


def numericGrad(X, Y, Ws, bs, funs, cost, forwardProp, epsilon = 0.1, **kwargs):
	encode = lambda Ws: np.zeros((0,1)) if Ws == [] else np.append(Ws[0].reshape(-1,1), encode(Ws[1:]), axis = 0)
	decode = lambda Wall, shapes: [] if shapes == [] else [Wall[:np.prod(shapes[0]),:].reshape(shapes[0])] + decode(Wall[np.prod(shapes[0]):,:] , shapes[1:])

	shapesW = list(map(lambda x: x.shape, Ws))
	shapesb = list(map(lambda x: x.shape, bs))
	Wall = encode(Ws)
	ball = encode(bs)

	dw = np.eye(Wall.shape[0])*epsilon
	db = np.eye(ball.shape[0])*epsilon

	W1 = Wall + dw
	b1 = ball + db

	W2 = Wall - dw
	b2 = ball - db

	dW = np.zeros(shape = Wall.shape)
	db = np.zeros(shape = ball.shape)

	for i in range(W1.shape[0]):
		A1 = forwardProp(X, decode(W1[:,i:i+1], shapesW), bs, funs)[-1][1]
		A2 = forwardProp(X, decode(W2[:,i:i+1], shapesW), bs, funs)[-1][1]

		dW[i,0] = (np.sum(cost(A1, Y, A1.shape[1])) - np.sum(cost(A2, Y, A2.shape[1])))/(epsilon)

	for i in range(b1.shape[0]):
		A1 = forwardProp(X, Ws, decode(b1[:, i:i+1], shapesb), funs)[-1][1]
		A2 = forwardProp(X, Ws, decode(b2[:, i:i+1], shapesb), funs)[-1][1]

		db[i,0] = (np.sum(cost(A1, Y, A1.shape[1])) - np.sum(cost(A2, Y, A2.shape[1])))/(epsilon)

	return decode(dW, shapesW), decode(db, shapesb)
