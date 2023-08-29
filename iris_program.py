# Import statements

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint

# Columns

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Loading the data

df = pd.read_csv('iris.csv', names=columns)

# Showing the data

pprint(df.head())
pprint(df.describe())