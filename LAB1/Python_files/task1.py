import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats
from func import *

# Загружаю датасеты
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

socr_df = pd.read_csv("baseball.csv")
print(socr_df.head())

# Диаграммы Тьюки
draw_tukey_diag(
    iris_df, 
    'target', 
    'petal length (cm)', 
    'Диаграмма Тьюки (iris)'
)
plt.show()

draw_tukey_diag(
    socr_df, 
    'Height(inches)', 
    'Weight(pounds)', 
    'Диаграмма Тьюки (socr_df)'
)
plt.show()

# Корреляция + heatmap
draw_heatmap_corr(iris_df, "Iris corr heatmap")
plt.show()

draw_heatmap_corr(socr_df, "SOCR corr heatmap")
plt.show()

# Матрица частных коэф корреляции + проверка распределения
partial_iris_df = iris_df.pcorr()
partial_socr_df = socr_df.pcorr()

draw_partial_corr(partial_iris_df, 'Partial corr iris heatmap')
plt.show()

draw_partial_corr(partial_socr_df, 'Heatmap Partial socr_df')
plt.show()

# KC
ks_results = iris_df.apply(check_normality)
print(ks_results)

