import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats

from func import *
from task1 import (
    iris_df
)
from task2 import (
    r2_pair_iris,
    r2_multi_iris
)

clean_iris = remove_outliers_iqr(iris_df, 'sepal width (cm)')

x_train_pair, x_test_pair, y_train_pair, y_test_pair = train_test_split(
    clean_iris[["target"]],
    clean_iris["sepal length (cm)"],
    test_size=0.2,
    random_state=1
)

x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(
    clean_iris[["target", 'sepal width (cm)']],
    clean_iris["sepal length (cm)"],
    test_size=0.2,
    random_state=1
)

pair_model = LinearRegression().fit(x_train_pair, y_train_pair)
multi_model = LinearRegression().fit(x_train_multi, y_train_multi)

r2_pair_test = pair_model.score(x_test_pair, y_test_pair)
r2_multi_test = multi_model.score(x_test_multi, y_test_multi)

print("\n FOR IRIS WITH OUTLIERS:")
print(f'R^2 pair: {r2_pair_iris:.4f}')
print(f'R^2 multi: {r2_multi_iris:.4f}')
print(" FOR IRIS WITH NO OUTLIERS:")
print(f'R^2 pair: {r2_pair_test:.4f}')
print(f'R^2 multi: {r2_multi_test:.4f}')
