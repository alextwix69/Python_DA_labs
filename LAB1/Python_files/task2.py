import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

from func import *
from task1 import (
    iris_df,
    socr_df
)

""" PAIR test-train DATA"""
# socr
x_train_pair_reg_socr, x_test_pair_reg_socr, y_train_pair_reg_socr, y_test_pair_reg_socr = train_test_split(
    socr_df[["Weight(pounds)"]],
    socr_df["Height(inches)"],
    test_size=0.2,
    random_state=1
)
# iris
x_train_pair_reg_iris, x_test_pair_reg_iris, y_train_pair_reg_iris, y_test_pair_reg_iris = train_test_split(
    iris_df[["target"]],
    iris_df["sepal length (cm)"],
    test_size=0.2,
    random_state=1
)

""" MULTI test-train DATA """
# socr
x_train_multi_reg_socr, x_test_multi_reg_socr, y_train_multi_reg_socr, y_test_multi_reg_socr = train_test_split(
    socr_df[["Weight(pounds)", "Age"]],
    socr_df["Height(inches)"],
    test_size=0.2,
    random_state=1
)
# iris
x_train_multi_reg_iris, x_test_multi_reg_iris, y_train_multi_reg_iris, y_test_multi_reg_iris = train_test_split(
    iris_df[["target", "sepal width (cm)"]],
    iris_df["sepal length (cm)"],
    test_size=0.2,
    random_state=1
)

""" MODELS """
# pair
pair_reg_socr_model = LinearRegression().fit(
    x_train_pair_reg_socr,
    y_train_pair_reg_socr
)
pair_reg_iris_model = LinearRegression().fit(
    x_train_pair_reg_iris,
    y_train_pair_reg_iris
)

# multi
multi_reg_socr_model = LinearRegression().fit(
    x_train_multi_reg_socr,
    y_train_multi_reg_socr
)
multi_reg_iris_model = LinearRegression().fit(
    x_train_multi_reg_iris,
    y_train_multi_reg_iris
)

""" PREDICTIONS """
# pair
pair_reg_iris_pred = pair_reg_iris_model.predict(x_train_pair_reg_iris)
pair_reg_socr_pred = pair_reg_socr_model.predict(x_train_pair_reg_socr)

# multi
multi_reg_iris_pred = multi_reg_iris_model.predict(x_train_multi_reg_iris)
multi_reg_socr_pred = multi_reg_socr_model.predict(x_train_multi_reg_socr)

""" RESIDUALS """
# pair
pair_reg_iris_residuals = y_train_pair_reg_iris - pair_reg_iris_pred
pair_reg_socr_residuals = y_train_pair_reg_socr - pair_reg_socr_pred

# multi
multi_reg_iris_residuals = y_train_multi_reg_iris - multi_reg_iris_pred
multi_reg_socr_residuals = y_train_multi_reg_socr - multi_reg_socr_pred

""" R2 """
# pair
r2_pair_socr = pair_reg_socr_model.score(
    x_test_pair_reg_socr,
    y_test_pair_reg_socr
)
r2_pair_iris = pair_reg_iris_model.score(
    x_test_pair_reg_iris,
    y_test_pair_reg_iris
)

#multi
r2_multi_socr = multi_reg_socr_model.score(
    x_test_multi_reg_socr,
    y_test_multi_reg_socr
)
r2_multi_iris = multi_reg_iris_model.score(
    x_test_multi_reg_iris,
    y_test_multi_reg_iris
)


if r2_pair_iris < r2_multi_iris:
    iris_reg_res = "Multi regression is better than pair regression"
else:
    iris_reg_res = "Pair regression is better than multi regression"

if r2_pair_socr < r2_multi_socr:
    socr_reg_res = "Multi regression is better than pair regression"
else:
    socr_reg_res = "Pair regression is better than multi regression"


"""PREDICTIONS ON TEST DATA"""
#iris
pair_reg_iris_pred_test = pair_reg_iris_model.predict(x_test_pair_reg_iris)
multi_reg_iris_pred_test = multi_reg_iris_model.predict(x_test_multi_reg_iris)
#socr
pair_reg_socr_pred_test = pair_reg_socr_model.predict(x_test_pair_reg_socr)
multi_reg_socr_pred_test = multi_reg_socr_model.predict(x_test_multi_reg_socr)


""" PLOTS """
# iris
plotic(
    y_test_pair_reg_iris,
    pair_reg_iris_pred_test,
    y_test_multi_reg_iris,
    multi_reg_iris_pred_test
)
plt.show()

# socr
plotic(
    y_test_pair_reg_socr,
    pair_reg_socr_pred_test,
    y_test_multi_reg_socr,
    multi_reg_socr_pred_test
)
plt.show()

print('\n SOCR:')
print(f'multiple correlation = {np.sqrt(r2_multi_socr):.4f}')
print(f'standart error = {socr_df["Height(inches)"].sem():.4f}')
print(f'corrected pair R**2 = {(1 - (1 - r2_pair_socr) * 207 / 206):.4f}')
print(f'corrected multi R**2 = {(1 - (1 - r2_multi_socr) * 207 / 205):.4f}')
print(socr_reg_res)

print(f'\n IRIS:')
print(f'multiple correlation = {np.sqrt(r2_multi_iris):.4f}')
print(f'standart error = {iris_df["sepal length (cm)"].sem():.4f}')
print(f'corrected pair R**2 = {(1 - (1 - r2_pair_iris) * 29 / 28):.4f}')
print(f'corrected multi R**2 = {(1 - (1 - r2_multi_iris) * 29 / 27):.4f}')
print(iris_reg_res)