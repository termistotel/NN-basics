import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.framework import ops
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

composition = lambda a, b: (lambda x: a(b(x)))

class CrossEntropy(keras.losses.Loss):
	def call(self, y_true, y_pred):
		print(y_true, y_pred)
		cross_entropy = -tf.reduce_sum( y_true*tf.math.log(y_pred + 0.001), axis=[-1])
		loss = tf.reduce_mean(cross_entropy)
		return loss

def lossss (y_true, y_pred):
	cross_entropy = -tf.reduce_sum( y_true*tf.math.log(y_pred + 0.001), axis=[-1])
	loss = tf.reduce_mean(cross_entropy)
	return loss


class Iris(keras.Model):
	def __init__(self):
		super(Iris, self).__init__()

		f1 = keras.layers.Flatten()
		l1 = keras.layers.Dense(100, activation = tf.nn.relu)
		l2 = keras.layers.Dense(10, activation = tf.nn.softmax)
		self.lays = [l2,l1,f1]

	def call(self, inputs, training = False):
		out = reduce(composition, self.lays)(inputs)
		print("training: ", training)
		# g = tf.get_default_graph()
		# if training:
		# 	tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=100000)
		# 	tf.keras.backend.get_sesssion().run(tf.global_variables_initializer())
		# else:
		# 	tf.contrib.quantize.create_eval_graph(input_graph=g)

		return out


bsize = 128

# data = load_iris()
# x = (data.data*10).astype(np.float32)
# y = pd.get_dummies(data.target).values.astype(np.float32)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)
y_train = pd.get_dummies(y_train).values.astype(np.float32)
y_test = pd.get_dummies(y_test).values.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(150).batch(bsize)

evalset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
evalset = evalset.batch(bsize)

model = Iris()


model.compile(
	tf.optimizers.Adam(learning_rate=1e-3),
	# loss=CrossEntropy(),
	loss = tf.keras.losses.categorical_crossentropy,
	metrics=['accuracy']
)


hist = model.fit(
	x=dataset,
	epochs=1,
	verbose=1,
	callbacks=None,
	validation_data=evalset,
	initial_epoch=0,
	validation_freq=1,
)

model.save("model/model.hp5")

def representative_dataset_gen():
	for _ in range(100):
		loc = np.random.randint(0,len(y_test)-1)
		image = x_test[loc:loc+1]
		yield [image]

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("model/model.hp5", custom_objects={'CrossEntropy': CrossEntropy, "lossss": lossss})
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

with open("converted_model.tflite", "wb") as file:
	file.write(converter.convert())