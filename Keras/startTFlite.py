import numpy as np
import pandas as pd
# from edgetpu.classification.engine import ClassificationEngine
from edgetpu.basic.basic_engine import BasicEngine

from sklearn.datasets import load_iris

data = load_iris()
x_train = (data.data*10).astype(np.float32)
y_train = pd.get_dummies(data.target).values.astype(np.float32)


eng = BasicEngine("converted_model_edgetpu.tflite")
print(eng)
print(eng.get_all_output_tensors_sizes(), eng.get_input_tensor_shape())

for i in range(150):
# inp = np.array([1,2,3]).astype(np.uint8)
	inp = x_train[i].astype(np.uint8)
	# inp = [30,30,30,30]
	print(inp, x_train[i], y_train[i])
	print(eng.RunInference(inp)[1])
