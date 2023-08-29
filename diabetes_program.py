# from sklearn.pydatasets import load_diabetes
# from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd

# Loading the data

df = pd.read_csv('diabetes.csv')

# Showing the data

pprint(df.head)
pprint(df.describe())