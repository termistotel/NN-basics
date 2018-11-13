import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
import pandas as pd

if __name__ == "__main__":

	data = load_iris()

	X = data.data
	Y = pd.get_dummies(data.target).values

	model = keras.Sequential()
	model.add(keras.layers.Dense(10, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
	model.add(keras.layers.Dense(3, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01)))
	model.compile(optimizer = tf.train.AdamOptimizer(0.05),
		loss = 'categorical_crossentropy',
		metrics = ['accuracy'])
	model.fit(X, Y, epochs=10, batch_size=32)