import numpy as np
from sklearn import datasets

def getXor():
	return (np.array([[0,0], [0,1], [1,1], [1,0]]).T, np.array([[0,1,1,0]]))

def getIris():
	iris = datasets.load_iris()
	labels = iris.target
	unique = np.unique(labels)
	Y = np.zeros(shape=(len(unique), labels.shape[0]))
	for i, val in enumerate(unique):
		vector = np.zeros(shape=(len(unique), 1))
		vector[i] = 1
		Y[:, labels==val] = vector

	returnx = iris.data.T
	returny = Y

	for i in range(5):
		returnx = np.append(returnx, returnx, axis=1)
		returny = np.append(returny, returny, axis=1)

	returnx, returny = shuffleData(returnx, returny)

	return returnx, returny

def shuffleData(X, Y, seed=None):
	np.random.seed(1337)
	a = np.arange(Y.shape[1])
	np.random.shuffle(a)

	return X[:, a], Y[:, a]

if __name__ == "__main__":
	Y = getIris()[0]
	a = np.arange(Y.shape[1])
	np.random.shuffle(a)

	print(Y[:,a[:3]])
	Y = Y[:, a]

	print(Y[:,:3])
	print(Y[:,a[:3]])
	print(a)
