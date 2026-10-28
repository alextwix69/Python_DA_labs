import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

from LAB1.Python_files.func import *
from LAB1.Python_files.task1 import (
    iris_df,
    socr_df
)
# PAIR train DATA
# socr
x_train_pair_reg_socr = socr_df[["Weight(pounds)"]]
y_train_pair_reg_socr = socr_df["Height(inches)"]
# iris
x_train_pair_reg_iris = iris_df[["target"]]
y_train_pair_reg_iris = iris_df["sepal length (cm)"]

# MULTI train DATA
# socr
x_train_multi_reg_socr = socr_df[["Weight(pounds)", "Age"]]
y_train_multi_reg_socr = socr_df["Height(inches)"]
# iris
x_train_multi_reg_iris = iris_df[["target", "sepal width (cm)"]]
y_train_multi_reg_iris = iris_df["sepal length (cm)"]


pair_reg_socr = LinearRegression().fit(
    x_train_pair_reg_socr,
    y_train_pair_reg_socr
)

pair_reg_iris = LinearRegression().fit(
    x_train_pair_reg_iris,
    y_train_pair_reg_iris
)

multi_reg_socr = LinearRegression().fit(
    x_train_multi_reg_socr,
    y_train_multi_reg_socr
)

multi_reg_iris = LinearRegression().fit(
    x_train_multi_reg_iris,
    y_train_multi_reg_iris
)


