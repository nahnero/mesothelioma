# Started Wed  4 Mar 00:27:41 2020 by nahnero. #

import pandas as pd
import numpy as np
from sklearn.model_selection       import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn                       import metrics
from sklearn.metrics               import classification_report

df = pd.read_csv('./data.csv')
y = df['class of diagnosis'] - 1
x = df.loc[:, df.columns != 'class of diagnosis']

iteraciones = 1000
precision0  = [None]*iteraciones
precision1  = [None]*iteraciones
cm = [None]*iteraciones
a  = [None]*iteraciones
b  = [None]*iteraciones
c  = [None]*iteraciones
d  = [None]*iteraciones

lda = LDA ();

for i in range (iteraciones):
    X_train, X_test, y_train, y_test = train_test_split (x, y, test_size = 0.2);
    shhhhh = lda.fit (X_train, y_train);
    y_true = y_test
    y_pred = lda.predict (X_test)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    precision0[i] = report_dict['0']['precision']
    precision1[i] = report_dict['1']['precision']
    cm[i] = metrics.confusion_matrix (y_test, y_pred)
    print ('\33[1;1H\33[2JIteration %d/%d' %(i+1, iteraciones))

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

#  39.19  6.60
#  14.50 4.70
