
import pandas as pd                # tables (DataFrames)
import matplotlib.pyplot as plt    # basic plotting
import seaborn as sns              # nicer statistical plots on top of matplotlib
import sklearn as skl              # machine learning library


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.shape       # (rows, columns)
train.head()      # the first 5 rows
train.info()      # each column's type and how many values are filled in
train.describe()  # count, mean, min, max and quartiles for every column