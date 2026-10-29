import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

from func import *
from task1 import (
    socr_df,
    iris_df
)
from task2 import (
    pair_reg_iris_residuals,
    pair_reg_socr_residuals,
    multi_reg_iris_residuals,
    multi_reg_socr_residuals
)

kstest_pair_iris_resid = kstest(pair_reg_iris_residuals)
kstest_multi_iris_resid = kstest(multi_reg_iris_residuals)
kstest_pair_socr_resid = kstest(pair_reg_socr_residuals)
kstest_multi_socr_resid = kstest(multi_reg_socr_residuals)

print("\n IRIS:")
print(
    f'Pair regression residuals distribution: {kstest_decision(kstest_pair_iris_resid)}' + 
    f' ({kstest_pair_iris_resid.pvalue:.4f})'
)
print(
    f'Multi regression residuals distribution: {kstest_decision(kstest_multi_iris_resid)}' + 
    f' ({kstest_multi_iris_resid.pvalue:.4f})'
)
print(" SOCR:")
print(
    f'Pair regression residuals distribution: {kstest_decision(kstest_pair_socr_resid)}' + 
    f' ({kstest_pair_socr_resid.pvalue:.4f})'
)
print(
    f'Multi regression residuals distribution: {kstest_decision(kstest_multi_socr_resid)}' + 
    f' ({kstest_multi_socr_resid.pvalue:.4f})'
)

# DW TEST
durb_pair_iris = durbin(iris_df,["target"], 'sepal length (cm)')
durb_multi_iris = durbin(iris_df, ["target", 'sepal width (cm)'], 'sepal length (cm)')
durb_pair_socr = durbin(socr_df, ['Weight(pounds)'], 'Height(inches)')
durb_multi_socr = durbin(socr_df, ['Weight(pounds)', 'Age'], 'Height(inches)')

print("\n IRIS DW-TEST")
print(f'Pair: {durb_pair_iris:.4f}')
print(f'Multi: {durb_multi_iris:.4f}')
print(" SOCR DW-TEST")
print(f'Pair: {durb_pair_socr:.4f}')
print(f'Multi: {durb_multi_socr:.4f}')