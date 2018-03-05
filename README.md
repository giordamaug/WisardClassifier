# WisardClassifier
Machine learning supervised method for classification using WiSARD<sup>rp</sup>

> Authors: Maurizio Giordano and Massimo De Gregorio
> - Istituto di Calcolo e Reti ad Alte Prestazioni (ICAR) - Consiglio Nazionale delle Ricerche (CNR) (Italy)
> - Istituto di Scienze Applicate e Sistemi Intelligenti "Eduardo Caianiello" (ISASI) - Consiglio Nazionale delle Ricerche (CNR) (Italy)

----------------------
Description
----------------------

WisardClassifier is a machine learning classifer implemented as an exntension module of
the scikit-learn package in Python.
As a consequence, to use WisardClassifier you need the following packages installed in your
Python environment:

1) Numpy

2) Scikit-Learn

WisardClassifier is based on the WiSARD C++ Library (included in the <code>wislib</code> directory).

----------------------
Compile source (Linux, Mac OSX)
----------------------

To run the code the following libraries are required:

2. CMake  2.8  (later version may also work)

3. C++ Compiler (tested only with GCC 5.x or later versions)

```
$ cd wislib
$ cmake .
$ make
```

This will produce the WiSARD library object <code>libwisard-cxx_static_3.0.<dllext></code> in the 
same folder. In addition, a sample C++ program using the library is also compiled for testing (see next section).

----------------------
WiSARD in C++
----------------------

An example of usage of WiSARD library in your C++ programs is the program
<code>test.cpp</code> which is compiled with the library and can be run by typing:

```
$ test
```

The source code of program <code>test.cpp</code> is hereafter reported as an example of
use of WiSARD in C++ programming:


```cpp
//
//  test.cpp
//
//
//  Created by Maurizio Giordano on 20/03/2014
//
// the WISARD C++ implementation
//

#include "wisard.hpp"
#include <iostream>
#include <string>

unsigned int *mk_tuple(discr_t *discr, int *sample) {
    int i,j;
    /* alloc tuple array */
    unsigned int *intuple = (unsigned int *)malloc(discr->n_ram * sizeof(unsigned int));
    int x;
    for (i = 0; i < discr->n_ram; i++)
        for (j = 0; j < discr->n_bit; j++) {
            x = discr->map[(i * discr->n_bit) + j] % discr->size;
            intuple[i] += (1<<(discr->n_bit -1 - j))  * sample[x];
    }
    return intuple;
}

int main() {
   
    int X[8][8] ={{0, 1, 0, 0, 0, 0, 0, 0},
                  {0, 0, 1, 1, 1, 1, 0, 0},
                  {0, 0, 1, 0, 0, 0, 1, 0},
                  {1, 0, 0, 0, 0, 0, 0, 1},
                  {1, 1, 0, 1, 1, 1, 1, 1},
                  {1, 0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 1, 0, 0, 1},
                  {1, 0, 0, 0, 0, 0, 0, 1}};
    std::string y[8] = {"A","A","B","B","A","A","B","A"};
    double responses[2];
    int s;
    int test[8] = {0, 0, 1, 0, 0, 0, 1, 0};
    
    // init WiSARD (create discriminator for each class "A" and "B")
    discr_t wisard[2];
    wisard[0] = *make_discr(2,8,"random",0);
    wisard[1] = *make_discr(2,8,"random",0);

    // train WiSARD
    for (s=0; s < 8; s++)
        if (y[s] == "A")
            train_discr(wisard,mk_tuple(wisard,X[s]));
        else
            train_discr(wisard+1,mk_tuple(wisard+1,X[s]));
    
    // print WiSARD state
    print_discr(wisard);
    print_discr(wisard+1);
    
    // predict by WiSARD
    responses[0] = classify_tuple(wisard->rams,wisard->n_ram,mk_tuple(wisard,test));
    responses[1] = classify_tuple((wisard+1)->rams,(wisard+1)->n_ram,mk_tuple(wisard+1,test));
    printf("A responds with score %.2f\n",responses[0]);
    printf("B responds with score %.2f\n",responses[1]);
}
```

----------------------
WiSARD in Python
----------------------

To use WiSARD in your Python scripts you need to have
python 2.7 (or later) installed on your system, plus the following
modules:

1. Numpy

2. Ctools

Once you have set the python programming framework, you can use the file <code>test.py</cose> simple
script to start using WiSARD.

```python
from wisardwrapper import *
import numpy as np

def mk_tuple(discr, sample):
    intuple = np.zeros(discr.contents.n_ram, dtype = np.uint64)
    for i in range(discr.contents.n_ram):
        for j in range(discr.contents.n_bit):
            x = discr.contents.map[(i * discr.contents.n_bit) + j]
            intuple[i] += (2**(discr.contents.n_bit -1 - j))  * sample[x]
    return intuple


X = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 0, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1]], np.int32)

y = np.array(["A","A","B","B","A","A","B","A"])

test = np.array([0, 0, 1, 0, 0, 0, 1, 0], np.int32)

# init WiSARD (create discriminator for each class "A" and "B")
wisard = {}
wisard["A"] = make_discr(2,8,"random",0)
wisard["B"] = make_discr(2,8,"random",0)

# train WiSARD
for s in range(X.shape[0]):
    tuple = mk_tuple(wisard[y[s]],X[s])
    train_discr(wisard[y[s]],tuple)

# print WiSARD state
print_discr(wisard["A"]);
print_discr(wisard["B"]);
    
# predict by WiSARD
responses = {}
test_tuple = mk_tuple(wisard["A"],test)
responses["A"] = classify_discr(wisard["A"],test_tuple);
test_tuple = mk_tuple(wisard["B"],test)
responses["B"] = classify_discr(wisard["B"],test_tuple);
print("A responds with score %.2f\n"%responses["A"]);
print("B responds with score %.2f\n"%responses["B"]);
```
----------------------
WiSARD in Scikit Learn (Python)
----------------------

Hereafter we report a Python script <code>test_wis.py</code> as an example of usage of WisardClassifier within the Scikit-Learn
machine learning programming framework. For a more complete example, see file <code>test.py</code>.

```
# import sklearn and scipy stuff
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
import scipy.sparse as sps
from scipy.io import arff
# import wisard classifier library
from wis import WisardClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from utilities import *
import time

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
```
