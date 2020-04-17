# Started Fri 17 Apr 01:09:16 2020 by nahnero. #

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./data.csv')

y = df['class of diagnosis'].replace (1, 0) #  sick -> 0
y = y.replace (2, 1)                        #  sick -> 1
x = df.loc[:, df.columns != 'class of diagnosis']

ignorecols = [
       'type of MM',
       'diagnosis method',
       'cytology',
       'dead or not',
       ]

x = x.drop (columns=ignorecols)
labels = x.columns;

#  PCA
from sklearn.preprocessing import StandardScaler

x = StandardScaler ().fit_transform (x)

from sklearn.decomposition import PCA

n_components = 20
pca = PCA (n_components = n_components)
principalComponents = pca.fit_transform(x)
evr = pca.explained_variance_ratio_

print (evr)
print (np.cumsum (evr))

import matplotlib as mpl
import matplotlib.pyplot as plt

#  Pareto
fig, ax = plt.subplots ()
ax.bar (range (len (evr)), evr)
ax.set_ylim (top=1)
ax.set_xticks (range (n_components))
ax.set_xticklabels (range (n_components))
ax1 = ax.twinx ()
ax1.set_ylim (top=100)
ax1.plot (range (len (evr)), np.cumsum (evr)*100, marker = '.', color = 'red')
fig.savefig  ('../images/pareto.pdf', bbox_inches = 'tight', pad_inches = 0)
