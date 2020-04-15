# Started Fri 10 Apr 18:33:39 2020 by nahnero. #

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./data.csv')

y = df['class of diagnosis'].replace (1, 0) #  sick -> 0
y = y.replace (2, 1)                        #  sick -> 1
x = df.loc[:, df.columns != 'class of diagnosis']

ignorecols = [
       'asbestos exposure',
       'type of MM',
       'duration of asbestos exposure',
       'diagnosis method',
       'cytology',
       ]

x = x.drop (columns=ignorecols)

#  from sklearn.preprocessing import StandardScaler
#  from sklearn.decomposition import KernelPCA

#  x = StandardScaler ().fit_transform (x) #  typify
#  pca = KernelPCA (n_components = 10, kernel = 'rbf')
#  pca.fit_transform(x)

#  ooooooooo.   ooooo      ooo ooooo      ooo
#  `888   `Y88. `888b.     `8' `888b.     `8'
#   888   .d88'  8 `88b.    8   8 `88b.    8
#   888ooo88P'   8   `88b.  8   8   `88b.  8
#   888          8     `88b.8   8     `88b.8
#   888          8       `888   8       `888
#  o888o        o8o        `8  o8o        `8

from neupy import algorithms

modeloPNN = algorithms.PNN (
        std=5,
        verbose=False,
        )

#  ooo        ooooo ooooo        ooooooooo.
#  `88.       .888' `888'        `888   `Y88.
#   888b     d'888   888          888   .d88'
#   8 Y88. .P  888   888          888ooo88P'
#   8  `888'   888   888          888
#   8    Y     888   888       o  888
#  o8o        o888o o888ooooood8 o888o

from sklearn.neural_network import MLPClassifier as MLP

modeloMLP = MLP(
        hidden_layer_sizes = (100, 50, 25, ),
        max_iter = 500,
        #  activation = 'relu',
        #  learning_rate = 'invscaling',
        random_state = 1)

modelos = {
    'PNN' : modeloPNN,
    'MLP' : modeloMLP,
    }

iteraciones = 500
precision0, precision1 = [[None]*iteraciones for i in range (2)]
cm, a, b, c, d = [[None]*iteraciones for i in range (5)]

chosen = 'PNN'
from tqdm import tqdm
for i in tqdm (range (iteraciones), ncols=70):
    X_train, X_test, y_train, y_test = train_test_split (x, y, test_size = 0.3);
    try:
        shhhhh = modelos[chosen].fit (X_train, y_train);
    except:
        pass
    y_true = y_test
    y_pred = modelos[chosen].predict (X_test)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    precision0[i] = report_dict['0']['precision']
    precision1[i] = report_dict['1']['precision']
    cm[i] = metrics.confusion_matrix (y_test, y_pred)

e0 = pd.DataFrame (data = precision0)
e1 = pd.DataFrame (data = precision1)

print ('\nVerdaderos Negativos')
print (e0.describe ())
print ('\nVerdaderos Positivos')
print (e1.describe ())

j = 0;
for i in cm:
    a[j] = i[0][0]
    b[j] = i[0][1]
    c[j] = i[1][0]
    d[j] = i[1][1]
    j = j + 1

print ('\nConfusion Matrix')
print ('\n%.2f  %.2f\n%.2f %.2f' %(np.mean (a), np.mean (b), np.mean (c), np.mean (d)))
print ('\n%.2f%%' %(100*(np.mean(a) + np.mean(d))/(np.mean(a) + np.mean(b) + np.mean(c) + np.mean(d))))
