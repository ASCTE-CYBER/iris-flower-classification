from sklearn.datasets import load_diabetes
from pprint import pprint
import pandas as pd
import numpy as np

data = load_diabetes()

diabetes = pd.read_csv('diabetes.csv')

pprint(diabetes.columns)

pprint(diabetes.head)