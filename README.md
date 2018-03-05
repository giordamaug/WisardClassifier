# # WisardClassifier
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

2) Scikit

3) Scikit-Learn

WisardClassifier is based on the WiSARD C++ Library, that can be download for installation at:
https://github.com/giordamaug/WisardLibrary (see the referring github page for installation of library).

Be sure that, once compiled, the library binary (libwisard-cxx-X.Y.<ext>) is in the same directory of the
python script. Otherwise you have to fix the library path in wrapper file <code>wisard.py</code>

----------------------
Docker Installation/Usage
----------------------

First, you need to have docker client installed on your system. Then, you can install and test WisardClassfier 
in a docker container with the command:

```bash
$ docker build -t gioma/wiscl:v1 .
```
To run a bash shell in a docker container runnin gon the built image, and then run the 
WisardClassifier <code>test.py</code> program script, use the following commands:

```bash
$ docker run -it gioma/wiscl:v1 bash
root@<imageID>:/home/WisardClassifier# python test.py
```

----------------------
Usage
----------------------

Hereafter we report a Python script as an example of usage of WisardClassifier within the Scikit-Learn
machine learning programming framework. For a more complete example, see file <code>test.py</code>.

```
# import sklearn and scipy stuff
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
import scipy.sparse as sps
# import wisard classifier library
from wis import WIS
from sklearn.metrics import confusion_matrix, accuracy_score
from utilities import *

# IRIS (libsvm) - load datasets
X_train, y_train = load_svmlight_file(open("iris.libsvm", "r"))
class_names = np.unique(y_train)
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data

# IRIS - cross validation example
clf = WIS(nobits=16,notics=1024,debug=True)
kf = cross_validation.StratifiedKFold(y_train, 10)
predicted = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf, n_jobs=1, verbose=0)
print("Accuracy Avg: %.2f" % predicted.mean())
