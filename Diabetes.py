from sklearn.datasets import load_diabetes # Data Set
from sklearn.model_selection import train_test_split # Data preprocessing for ML
from pprint import pprint # Nice format for printing

data = load_diabetes()

pprint(data.keys())