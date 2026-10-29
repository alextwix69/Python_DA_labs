import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats
from func import *

barotrop = pd.read_csv(r'csv\Barotrop.csv').corr()
chem = pd.read_csv(r'csv\Chemical_ process.csv').corr()
factor = pd.read_csv(r'csv\Factor.csv').corr()

draw_heatmap_corr(barotrop, 'Heatmap barotrop'); plt.show()
draw_heatmap_corr(chem,'Heatmap chemical'); plt.show()
draw_heatmap_corr(factor,'Heatmap factor'); plt.show()
