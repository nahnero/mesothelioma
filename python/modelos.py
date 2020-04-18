# Started Fri 17 Apr 02:08:09 2020 by nahnero. #

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
       'type of MM',
       'diagnosis method',
       'cytology',
       'dead or not',
       ]

x = x.drop (columns=ignorecols)
print (x.columns)
#  for i in x.loc[12]: print ("%g,"%(i), end = '')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

modeloLDA = LinearDiscriminantAnalysis ();

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

modeloQDA = QuadraticDiscriminantAnalysis ();

from sklearn.neighbors import KNeighborsClassifier

modeloKNN   = KNeighborsClassifier (
        n_neighbors = 50
        )

from sklearn.ensemble import RandomForestClassifier

modelFOREST = RandomForestClassifier (
        n_estimators = 100,
        criterion = 'gini',
        )

from neupy import algorithms

modeloPNN = algorithms.PNN (
        std=5,
        verbose=False,
        )

from sklearn.neural_network import MLPClassifier as MLP

modeloMLP = MLP(
        hidden_layer_sizes = (175, 100, 50, 25, ),
        max_iter = 500,
        random_state = 1)

from sklearn import svm

modeloSVM = svm.LinearSVC()

modelos = {
    'LDA'    : modeloLDA,
    'QDA'    : modeloQDA,
    'KNN'    : modeloKNN,
    'FOREST' : modelFOREST,
    'SVM'    : modeloSVM,
    'PNN'    : modeloPNN,
    'MLP'    : modeloMLP,
    }

iteraciones = 1000
precision0, precision1 = [[None]*iteraciones for i in range (2)]
cm, TN, FP, FN, TP = [[None]*iteraciones for i in range (5)]

print ('Model & TP & FP & FN & TN & Accuracy & Sensitivity & Specificity & Time\\\\')
import time
for chosen in modelos:
    start = time.time()
    for i in range (iteraciones):
        X_train, X_test, y_train, y_test = train_test_split (x, y, test_size = 0.3);
        try:
            shhhhh = modelos[chosen].fit (X_train, y_train);
        except:
            pass
        y_true = y_test
        y_pred = modelos[chosen].predict (X_test)
        cm[i] = metrics.confusion_matrix (y_test, y_pred)

    e0 = pd.DataFrame (data = precision0)
    e1 = pd.DataFrame (data = precision1)

    j = 0;
    for i in cm:
        TN[j] = i[0][0]
        FP[j] = i[0][1]
        FN[j] = i[1][0]
        TP[j] = i[1][1]
        j = j + 1

    accuracy = (np.mean(TN) + np.mean(TP))/(np.mean(TN)\
        + np.mean(FP) + np.mean(FN) + np.mean(TP))
    sensitivity = np.mean(TP)/(np.mean(FN) + np.mean(TP))
    specificity = np.mean(TN)/(np.mean(FN) + np.mean(TN))

    print ('%6s & %.2f & %.2f & %.2f & %.2f &  %.2f & %.2f & %.2f & %.2f\\\\'\
            %(chosen, np.mean (TN), np.mean (FP), np.mean (FN),\
                np.mean (TP), accuracy, sensitivity, specificity,\
                time.time()-start))
