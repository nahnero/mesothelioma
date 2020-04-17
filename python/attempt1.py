# Started Wed  4 Mar 00:27:41 2020 by nahnero. #

import pandas as pd
import numpy as np
from sklearn.metrics               import classification_report
from sklearn.model_selection       import train_test_split
from sklearn                       import metrics

df = pd.read_csv('./data.csv')
y = df['class of diagnosis'] - 1
x = df.loc[:, df.columns != 'class of diagnosis']

dr = [
       #  'age',
       #  'gender',
       #  'city',
       'asbestos exposure',
       'type of MM',
       'duration of asbestos exposure',
       'diagnosis method',
       #  'keep side',
       'cytology',
       #  'duration of symptoms',
       #  'dyspnoea',
       #  'ache on chest',
       #  'weakness',
       #  'habit of cigarette',
       #  'performance status',
       #  'white blood',
       #  'cell count (WBC)',
       #  'hemoglobin (HGB)',
       #  'platelet count (PLT)',
       #  'sedimentation',
       #  'blood lactic dehydrogenise (LDH)',
       #  'alkaline phosphatise (ALP)',
       #  'total protein',
       #  'albumin',
       #  'glucose',
       #  'pleural lactic dehydrogenise',
       #  'pleural protein',
       #  'pleural albumin',
       #  'pleural glucose',
       #  'dead or not',
       #  'pleural effusion',
       #  'pleural thickness on tomography',
       #  'pleural level of acidity (pH)',
       #  'C-reactive protein (CRP)'
       ]

#  x = x.drop (columns=['diagnosis method'])
x = x.drop (columns=dr)

#  PCA
#  from sklearn.preprocessing         import StandardScaler
#  from sklearn.decomposition         import PCA

#  x = StandardScaler ().fit_transform (x) #  typify
#  pca = PCA (n_components = 10)
#  principalComponents = pca.fit_transform(x)
#  evr = pca.explained_variance_ratio_
#  print (evr)
#  print (np.cumsum (evr))

iteraciones = 1000
precision0, precision1 = [[None]*iteraciones for i in range (2)]
cm, a, b, c, d = [[None]*iteraciones for i in range (5)]

#  from sklearn.discriminant_analysis import lineardiscriminantanalysis as lda
#  model = lda ();
#  from sklearn.discriminant_analysis import quadraticdiscriminantanalysis as qda
#  model = qda ();
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier (n_neighbors = 50)

from tqdm import tqdm
for i in tqdm (range (iteraciones), ncols=70):
    X_train, X_test, y_train, y_test = train_test_split (x, y, test_size = 0.2);
    try:
        shhhhh = model.fit (X_train, y_train);
    except:
        print ('💩')
    y_true = y_test
    y_pred = model.predict (X_test)
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
#  print ('\nGround Truth')
#  print (y_true.values)
#  print ('\nPredictions')
#  print (y_pred)
