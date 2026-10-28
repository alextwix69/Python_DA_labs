import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats


def draw_heatmap_corr(df, title):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title(title)

def draw_tukey_diag(df, x, y, title):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(title)

def draw_partial_corr(df, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, vmin=-1, vmax=1, cmap='coolwarm')
    plt.title(title)

def check_normality(col):
    data = col.dropna()
    stat, p_value = stats.kstest(
                        data, 
                        stats.norm.cdf, 
                        args=(
                            data.mean(), 
                            data.std(ddof=1)
                        )
                    )
    
    if p_value < 0.05:
        name = 'non-normal'
    else:
        name = 'normal'

    return pd.Series({
            'KS_stat': round(stat, 4), 
            'P_value': round(p_value, 4), 
            'Distribution': name
         })

def plotic(y_test_p, y_pred_p, y_test_m, y_pred_m):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ind_p = np.arange(len(y_test_p))
    ax[0, 0].scatter(ind_p, y_test_p, marker='o', label='Реальные значения', color='red')
    ax[0,0].set_title('Предсказания по порядку')
    ax[0,0].set_xlabel('Порядковый номер')
    ax[0,0].set_ylabel('Предсказанный вес')
    ax[0,1].scatter(ind_p,y_test_p - y_pred_p, marker='o', label='Остатки', color='blue')
    ax[0,1].scatter(ind_p, y_pred_p, marker='o', label='Предсказанные значение', color='green')
    ax[0,1].set_title('Остаток по порядку')
    ax[0,1].set_xlabel('Порядковый номер')
    ax[0,1].set_ylabel('Остаток')
    ind_m = np.arange(len(y_test_m))
    ax[1,0].scatter(ind_m, y_test_m, marker='o', label='Реальные значения', color='black')
    ax[1,0].set_title('Предсказания по порядку')
    ax[1,0].set_xlabel('Порядковый номер')
    ax[1,0].set_ylabel('Предсказанный вес')
    ax[1,1].scatter(ind_m, y_test_m - y_pred_m, marker='o', label='Остатки', color='yellow')
    ax[1,1].scatter(ind_m, y_pred_m, marker='o', label='Предсказанные значения', color='purple')
    ax[1,1].set_title('Остаток по порядку')
    ax[1,1].set_xlabel('Порядковый номер')
    ax[1,1].set_ylabel('Остаток')
    plt.tight_layout()
