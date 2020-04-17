# Started Thu 16 Apr 22:32:47 2020 by nahnero. #
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

import sklearn.feature_selection as sk

Fscore, pval = sk.f_classif (x, y)
r1 = Fscore.argsort().argsort()

print ("Ranking    Variable")
print ("-------    --------")
for i,j in enumerate (r1):
    print ("%2d         %s" %(i+1, labels[j]))

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector

#  model = KNeighborsClassifier (n_neighbors = 50)
model = RandomForestClassifier (n_estimators = 100, criterion = 'gini')

sfs = SequentialFeatureSelector (model,
                k_features = 10,
                forward = True,
                scoring = 'precision',
                cv = 10,
                )

sfs.fit (x, y, custom_feature_names = labels)
print (sfs.k_score_)
print ('Sequential Forward  Selection', sfs.k_feature_names_, end = '\n\n')

sfs.forward = False

sfs.fit (x, y, custom_feature_names = labels)
print (sfs.k_score_)
print ('Sequential Backward Selection', sfs.k_feature_names_, end = '\n\n')
