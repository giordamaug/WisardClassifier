#import numpy library
import numpy as np
# import sklearn and scipy stuff
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.sparse as sps
from scipy.io import arff
# import wisard classifier library
from wis import WisardClassifier
import time
#import utilities for matplot
from  utilities import *

# (Try) import matplot for graphics
try:
    import matplotlib.pyplot as plt
    matplotfound = True
except ImportError:
    matplotfound = False
    pass

B_enabled = True
# IRIS (arff) - load datasets
data, meta = arff.loadarff(open("datasets/iris.arff", "r"))
y_train = np.array(data['class'])
X_train = np.array([list(x) for x in data[meta._attrnames[0:-1]]])
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data
class_names = np.unique(y_train)
# IRIS (arff) - cross validation example
clf = WisardClassifier(n_bits=16,bleaching=B_enabled,n_tics=256,mapping='linear',debug=True,default_bleaching=3)
kf = cross_validation.LeaveOneOut(len(class_names))
predicted = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf, n_jobs=1)
print("Accuracy Avg: %.2f" % predicted.mean())

# IRIS (libsvm) - load datasets
X_train, y_train = load_svmlight_file(open("datasets/iris.libsvm", "r"))
class_names = np.unique(y_train)
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data
# IRIS - cross validation example (with fixed seed)
clf = WisardClassifier(n_bits=16,n_tics=1024,debug=True,bleaching=B_enabled,random_state=848484848)
kf = cross_validation.StratifiedKFold(y_train, 10)
predicted = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf, n_jobs=1, verbose=0)
print("Accuracy Avg: %.2f" % predicted.mean())

# DNA (libsvm) - load datasets
X_train, y_train = load_svmlight_file(open("datasets/dna.tr", "r"))
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data
class_names = np.unique(y_train)
X_test, y_test = load_svmlight_file(open("datasets/dna.t", "r"))
X_test = X_test.toarray() if sps.issparse(X_test) else X_test  # avoid sparse data

# DNA (arff) - testing example
clf = WisardClassifier(n_bits=16,n_tics=512,debug=True,bleaching=B_enabled,random_state=848484848,n_jobs=-1)
y_pred = clf.fit(X_train, y_train)
tm = time.time()
y_pred = clf.predict(X_test)
print("Time: %d"%(time.time()-tm))
predicted = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: %.2f" % predicted)

# DNA - plot (print) confusion matrix
if matplotfound:
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names,title='Confusion matrix')
    plt.show()
else:
    print_confmatrix(cm)

