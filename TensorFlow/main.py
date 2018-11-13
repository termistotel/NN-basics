import tensorflow as tf
import numpy as np

from sklearn.datasets import load_iris
import pandas as pd

if __name__ == "__main__":
	data = load_iris()
	X = data.data.T
	Y = pd.get_dummies(data.target).values.T

	arh = [X.shape[0]] + [10] + [Y.shape[0]]
	Wshapes = []
	for i in range(len(arh)-1):
		Wshapes.append((arh[i+1], arh[i]))

	acts = []
	for i in range(len(arh)-2):
		acts.append(tf.nn.relu)
	acts.append(lambda x: tf.nn.softmax(x, axis=0))


	graph1 = tf.Graph()

	with graph1.as_default():
		Ws = list(map(lambda x: tf.Variable(tf.random.normal(shape=x, dtype=tf.float32), dtype=tf.float32), Wshapes))
		bs = list(map(lambda x: tf.Variable(tf.random.normal(shape=(x,1), dtype=tf.float32), dtype=tf.float32), arh[1:]))
		# x = tf.placeholder(dtype=tf.float32, shape=(arh[0], None), name="x")
		# y = tf.placeholder(dtype=tf.float32, shape=(arh[-1], None), name="y")
		x = tf.constant(X, dtype=tf.float32)
		y = tf.constant(Y, dtype=tf.float32)
		iteration = tf.Variable(0.0)

		f = x
		for W, b, act in zip(Ws, bs, acts):
			f = act(tf.matmul(W, f) + b)

		loss = tf.losses.log_loss(y, f)
		lrExpDec = 0.15*0.95**(iteration/100)

		optimize = tf.train.AdamOptimizer(learning_rate=lrExpDec).minimize(loss, global_step = iteration)

	with tf.Session(graph=graph1) as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(1000):
			# print(X.shape)
			# print(f)
			# print(sess.run(f))
			print(sess.run(loss))
			sess.run(optimize, feed_dict={x: X, y: Y})

		print(np.sum(np.square(sess.run(f) - sess.run(y))))
