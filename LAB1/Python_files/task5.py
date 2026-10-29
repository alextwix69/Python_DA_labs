import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import root_mean_squared_error
from scipy import stats

from func import *
from task2 import (
    y_test_multi_reg_iris,
    y_test_multi_reg_socr,
    multi_reg_iris_pred_test,
    multi_reg_socr_pred_test,
    multi_reg_socr_model,
    multi_reg_iris_model,
    x_test_multi_reg_iris,
    x_test_multi_reg_socr
)

sko_iris_test = root_mean_squared_error(
    y_test_multi_reg_iris, 
    multi_reg_iris_pred_test
)

sko_socr_test = root_mean_squared_error(
    y_test_multi_reg_socr, 
    multi_reg_socr_pred_test
)

r2_iris_test = multi_reg_iris_model.score(
    x_test_multi_reg_iris,
    y_test_multi_reg_iris
)

r2_socr_test = multi_reg_socr_model.score(
    x_test_multi_reg_socr,
    y_test_multi_reg_socr
)

# ERROR PLOT

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
datasets = [
    ("Iris", y_test_multi_reg_iris, multi_reg_iris_pred_test),
    ("SOCR", y_test_multi_reg_socr, multi_reg_socr_pred_test)
]

for ax, (name, y_test, y_pred) in zip(axes, datasets):
    residuals = y_test.to_numpy() - y_pred

    ax.scatter(y_pred, residuals, alpha=0.7)

    ax.set_title(f"{name}: график остатков")
    ax.set_xlabel("Предсказанные значения")
    ax.set_ylabel("Остатки")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n IRIS")
print(f'SKO: {sko_iris_test:.4f}')
print(f'R^2: {r2_iris_test:.4f}')
print(" SOCR")
print(f'SKO: {sko_socr_test:.4f}')
print(f'R^2: {r2_socr_test:.4f}')



