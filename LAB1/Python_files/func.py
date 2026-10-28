import pandas as pd
import matplotlib.pyplot as plt
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
