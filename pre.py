# Started Wed  4 Mar 00:27:41 2020 by nahnero. #

import pandas as pd
import numpy as np
from sklearn.metrics               import classification_report
from sklearn.model_selection       import cross_val_predict, KFold
from sklearn.model_selection       import cross_val_score
from sklearn.metrics               import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

df = pd.read_csv('./data.csv')
y = df['class of diagnosis'] - 1
x = df.loc[:, df.columns != 'class of diagnosis']

print (x.columns)
#  for i in x.columns:
    #  print (i)

#  x = x.drop (columns=['dead or not'])

import matplotlib.pyplot as plt
import seaborn as sns

# Primero eliminamos variables muy correladas con el resto

from sklearn.preprocessing         import StandardScaler
from sklearn.decomposition         import PCA

#  df = StandardScaler ().fit_transform (df) #  typify
#  df = pd.DataFrame(data=df)
#  # dibujamos las correlaciones
#  corr = df.corr()
#  mask = np.triu(np.ones_like(corr, dtype=np.bool))
#  f, ax = plt.subplots(figsize = (17, 17))
#  cmap = sns.diverging_palette(200, 10, as_cmap=True)
#  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            #  square=True, linewidths=.1, cbar_kws={"shrink": .5})
#  b, t = plt.ylim() # discover the values for bottom and top
#  b += 0.5 # Add 0.5 to the bottom
#  t -= 0.5 # Subtract 0.5 from the top
#  plt.ylim(b, t) # update the ylim(bottom, top) values

#  f.savefig  ('./images/corrtodasnorm.pdf', bbox_inches = 'tight')

# calculamos la correlacion media
#  mean = np.mean (corr)
# seleccionamos las variables con mas correlacion (percentil 90)
#  todrop = mean[mean > np.percentile (mean, 90)]
# las eliminamos
#  x = x.drop (columns=todrop.index)

# dibujamos las correlaciones otra vez
#  corr = x.corr()
#  mask = np.triu(np.ones_like(corr, dtype=np.bool))
#  f, ax = plt.subplots(figsize = (17, 17))
#  cmap = sns.diverging_palette(200, 10, as_cmap=True)
#  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
#              square=True, linewidths=.1, cbar_kws={"shrink": .5})
#  f.savefig  ('./images/corrdesp.pdf', bbox_inches = 'tight')

#  PCA
from sklearn.preprocessing         import StandardScaler
#  from sklearn.decomposition         import PCA

x = StandardScaler ().fit_transform (x) #  typify
#  pca = PCA (n_components = 1)
#  principalComponents = pca.fit_transform(x)
#  evr = pca.explained_variance_ratio_

# Pareto
#  fig, ax = plt.subplots ()
#  ax.bar (range (len (evr)), evr)
#  ax.set_ylim (top=1)
#  ax1 = ax.twinx ()
#  ax1.set_ylim (top=100)
#  ax1.plot (range (len (evr)), np.cumsum (evr)*100, marker = '.', color = 'red')
#  fig.suptitle ('Pareto', fontsize = 12)
#  fig.savefig  ('./images/pareto.pdf', bbox_inches = 'tight', pad_inches = 0)

#  Test

"""
from sklearn.model_selection       import train_test_split
from sklearn                       import metrics
#  print (report_dict)

iteraciones = 1
precision0  = [None]*iteraciones
precision1  = [None]*iteraciones
cm = [None]*iteraciones
a  = [None]*iteraciones
b  = [None]*iteraciones
c  = [None]*iteraciones
d  = [None]*iteraciones

lda = LDA ();
from tqdm import tqdm
for i in tqdm (range (iteraciones), ncols=70):
    X_train, X_test, y_train, y_test = train_test_split (x, y, test_size = 0.7);
    try:
        shhhhh = lda.fit (X_train, y_train);
    except:
        print ('💩')
    y_true = y_test
    y_pred = lda.predict (X_test)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    precision0[i] = report_dict['0']['precision']
    precision1[i] = report_dict['1']['precision']
    cm[i] = metrics.confusion_matrix (y_test, y_pred)
    #  print ('\33[1;1H\33[2JIteration %d/%d' %(i+1, iteraciones))

e0 = pd.DataFrame (data = precision0)
e1 = pd.DataFrame (data = precision1)
print (e0.describe ())
print (e1.describe ())

j = 0;
for i in cm:
    a[j] = i[0][0]
    b[j] = i[0][1]
    c[j] = i[1][0]
    d[j] = i[1][1]
    j = j + 1

print ('\n%.2f  %.2f\n%.2f %.2f' %(np.mean (a), np.mean (b), np.mean (c), np.mean (d)))
print (y_true.values)
print (y_pred)
"""
