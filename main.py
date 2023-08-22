from chillml import Network
from chillml.layers import FullyConnected, Activation
from chillml.activations import Sigmoid
from chillml.losses import MeanSquaredError
import numpy as np
import pandas as pd

data = pd.read_csv('IRIS.csv')
data = data.sample(frac=1, ignore_index=True)

# sepal_length sepal_width petal_length petal_width

data['species'] = data['species'].replace({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}).apply(lambda x: np.array([[x]]))

training_inputs = data.iloc[:, :4].values.tolist()
training_inputs = [np.array([input]) for input in training_inputs]
training_outputs = data['species'].to_list()

layers = [
  FullyConnected(4, 6),
  Activation(Sigmoid),
  FullyConnected(6, 3),
  Activation(Sigmoid),
  FullyConnected(3, 1)
]

network = Network(layers, MeanSquaredError)

network.train(training_inputs, training_outputs, 1000, 0.2)