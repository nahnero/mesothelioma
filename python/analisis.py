# Started Thu 16 Apr 15:31:05 2020 by nahnero. #

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
#  print (x.info ())

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots (nrows = 6, ncols = 5, figsize = (17, 20))
ax = ax.flatten ()

#  Boxplot
#  for i in range (0, 30):
#      ax[i].boxplot ([x [labels[i]][y == 0], x [labels[i]][y == 1]])
#      ax[i].title.set_text (labels [i])
#      ax[i].set_xticklabels (['sano', 'enfermo'])

#  fig.savefig  ('../images/boxplot.pdf', bbox_inches = 'tight', pad_inches = 0)

# Histogram
#  for i in range (0, 30):
#      ax[i].hist ([x [labels[i]][y == 0], x [labels[i]][y == 1]],\
#              label = ['sano', 'enfermo'])
#      ax[i].title.set_text (labels [i])
#      ax[i].legend (loc = 1, prop={'size': 10})
#  fig.savefig  ('../images/histogram.pdf', bbox_inches = 'tight', pad_inches = 0)

# Kernel Density
#  from scipy.stats import gaussian_kde
#  for i in range (0, 30):
#      x_ = x [labels[i]][y == 0]
#      kde = gaussian_kde (x_)
#      xs = np.linspace(np.min (x_) - 10, np.max (x_), num=len (x_))

#      x1_ = x [labels[i]][y == 1]
#      kde1 = gaussian_kde (x1_)
#      xs1 = np.linspace(np.min (x1_) - 10, np.max (x1_), num=len (x1_))

#      ax[i].plot (xs, kde(xs), label = 'sano')
#      ax[i].plot (xs1, kde1(xs1), label = 'enfermo')
#      ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#      ax[i].legend (loc = 1, prop={'size': 10})

#  fig.savefig  ('../images/kdens.pdf', bbox_inches = 'tight', pad_inches = 0)

# QQPlot
import statsmodels.api as sm
for i in range (0, 30):
    sm.qqplot (x [labels[i]][y == 0], ax = ax[i], line = 's',\
            label = 'sano', c = (0, 0, 1, 0.6))
    sm.qqplot (x [labels[i]][y == 1], ax = ax[i], line = 's',\
            label = 'enfermo', c = (1,0.647059,0, 0.6))
    ax[i].title.set_text (labels [i])
    ax[i].set_xlabel ('')
    ax[i].set_ylabel ('')
    #  ax[i].axes.get_xaxis().set_visible(False)
    #  ax[i].axes.get_yaxis().set_visible(False)
    ax[i].legend (loc = 1, prop={'size': 10})

fig.savefig  ('../images/qqplot.pdf', bbox_inches = 'tight', pad_inches = 0)
