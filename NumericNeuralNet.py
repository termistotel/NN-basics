import numpy as np

# Thanks to NextGreen for most of the functions and classes below

class Activationfunction:
	def nofunction (x):
		return x
	def binarystep(x):
		if x<0:
			y=0
		else: y=1
		return y
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	def tanh(x):
		return (2/(1+np.exp(-2*x)))-1
	def ReLU(x):
		return x.clip (0, out=x)
	def leakyReLU(x):
		return np.where (x<0, x*0.01, x)

class Errorfunction:
	def meanSquaredError (neurons,result):
		out=0
		for mjerenje,i in enumerate(neurons):
			out+=(0.5*((result[mjerenje]-neurons[mjerenje])**2).sum(axis=0))
		return out/len(neurons)

	def cost(neurons, result, M):
		mjerenja_u_redu=len(neurons.shape)-2
		if 1 in neurons:
			neurons=neurons-np.isin(neurons,1)*0.0000001
		if 0 in neurons:
			neurons=neurons+np.isin(neurons,0)*0.0000001
		out = -result*np.log(neurons)/M - (1-result)*np.log(1-neurons)/M

		return np.sum(out)
	
	def cost2(neurons, result, M):
		mjerenja_u_redu=len(neurons.shape)-2
		if 1 in neurons:
			neurons=neurons-np.isin(neurons,1)*0.0000001
		if 0 in neurons:
			neurons=neurons+np.isin(neurons,0)*0.0000001

		out = -result*np.log(neurons)/M - (1-result)*np.log(1-neurons)/M

		while out.ndim > 1:
			out = np.sum(out, axis=1)

		return out

def randomInit(arhitecture):
	b=[]
	w=[]
	for j,i in enumerate (arhitecture):
		if not(j==0):
			b.append (np.random.random (i))
		if not (j==(len(arhitecture)-1)):
			w.append (np.random.random ((arhitecture [j+1],i)))
	return b,w

def tensortovector (input):
	max=0
	count=0
	for i in input:
		max+=i.size
	output=np.zeros (max)
	for i in input:
		output [count:(count+i.size)]=i.flatten ()
		count+=i.size
	return output

def forwardProp(input, Ws,bs, activation):
	trenutni = input
	for W,b in zip(Ws,bs):
		trenutni = activation(W.dot(trenutni) + b.reshape(-1,1))
	return trenutni

def diferentials(Ws, bs, dw):
	W = tensortovector(Ws)
	b = tensortovector(bs)

	Wsize = W.shape[0]
	bsize = b.shape[0]

	WB = np.hstack((W,b))

	DWB = WB + np.eye(WB.shape[0])*dw

	# print(DWB[:,0:Wsize])
	# print()
	# print(DWB[:,Wsize:Wsize+bsize])

	return DWB[:,0:Wsize], DWB[:,Wsize:Wsize+bsize], W, b

def derivation(input, ys, Ws, bs, dw=0.1, alfa=0.1, activation=Activationfunction.sigmoid, errorfun1=Errorfunction.cost, errorfun2=Errorfunction.cost2):
	WDW, bDb, W, b = diferentials(Ws, bs, dw)
	pomaknuti = input
	fun = lambda x: (x[0].shape, x[1].shape)
	tmp1 = 0
	tmp2 = 0
	M = input.shape[input.ndim-1]

	for i,j in map(fun, zip(Ws, bs)):
		tmpW = WDW[:,tmp1:tmp1+np.prod(i)]
		tmpb = bDb[:,tmp2:tmp2+np.prod(j)]
		newW = tmpW.reshape(-1, i[0], i[1])
		newb = tmpb.reshape(-1, j[0],1)	

		# print(newW.shape, pomaknuti.shape, newb.shape)

		if tmp1==0:
			pomaknuti = activation(newW.dot(pomaknuti) + newb)
		else:
			pomaknuti1 = np.zeros(shape=(newW.shape[0], newW.shape[1], pomaknuti.shape[2]))
			for i in range(newW.shape[0]):
				pomaknuti1[i,:,:] = activation(newW[i,:,:].dot(pomaknuti[i,:,:]) + newb[i,:,:])
			pomaknuti = pomaknuti1

		tmp1 += np.prod(i)
		tmp2 += np.prod(j)

	trenutni = forwardProp(input, Ws, bs, activation)
	# print("error: ", errorfun1(trenutni, ys, M))
	# print()

	# for i in range(pomaknuti.shape[0]):
	# 	print(trenutni, pomaknuti[i,:])

	df = (errorfun2(pomaknuti, ys,M) - errorfun1(trenutni, ys, M))/dw

	# print("df: ", df)

	Wsize = np.prod(W.shape)
	bsize = np.prod(b.shape)
	W -= alfa*df[:np.prod(W.shape)]
	b -= alfa*df[:np.prod(b.shape)]

	Ws1 = []
	bs1 = []
	tmp1 = 0
	tmp2 = 0

	for i,j in map(fun, zip(Ws, bs)):
		Ws1.append(W[tmp1:tmp1+np.prod(i)].reshape(i))
		bs1.append(b[tmp2:tmp2+np.prod(j)])

		tmp1+= np.prod(i)
		tmp2+= np.prod(j)

	return Ws1,bs1

if __name__ == '__main__':

	X = np.array([[1,0], [0,1], [0,0], [1,1]])
	y = np.array([[1,1,0,0]])
	arh=[2,10,1]
	w, b = randomInit(arh)

	np.set_printoptions(precision=1, suppress=1)

	for i in range(1000):
		if i%10 == 0:
			Mjer = forwardProp(X.T, w, b, Activationfunction.sigmoid)
			print("Mjerenje :", Mjer, y)
			print("Cost :", Errorfunction.cost(Mjer, y, y.size))
			print()

		w, b = derivation(X.T, y, w, b, dw=0.0000001, alfa=10)

