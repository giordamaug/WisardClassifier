"""
    WiSARD Classifier in Scikit-Learn Python Package

    Created by Maurizio Giordano on 13/12/2016

"""
import sys,os
import numpy as np
from numpy.core.multiarray import int_asbuffer
import ctypes
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse as sps
from wisardwrapper import *
import time
import random
import multiprocessing as mp
import multiprocessing.sharedctypes as mpsh
import warnings

mypowers = 2**np.arange(32, dtype = np.uint32)[::]
print mypowers[16]

# UTILITY
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[0;32m'
    WHITEBLACK = '\033[1m\033[40;37m'
    BLUEBLACK = '\033[1m\033[40;94m'
    YELLOWBLACK = '\033[1m\033[40;93m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def printProgressBar(label,time,etime,basecolor, cursorcolor, linecnt,progress,size):
    barwidth = 70
    progress = linecnt / float(size);
    str = '%s |' % label
    pos = int(barwidth * progress)
    str += basecolor
    for p in range(barwidth):
        if p < pos:
            str += u'\u2588'
        elif p == pos:
            str += color.END + cursorcolor + u'\u2588' + color.END + basecolor
        else:
            str += u'\u2591'
    str += color.END + '| ' + "{:>3}".format(int(progress * 100.0)) + ' % ' + color.YELLOWBLACK + ' ' + etime + ' ' + color.WHITEBLACK + time + ' ' + color.END
    sys.stdout.write("\r%s" % str.encode('utf-8'))
    sys.stdout.flush()
    return progress

def compTime(deltatime,progress):
    hours, rem = divmod(deltatime*((1.0-progress) / progress), 3600)
    hourse, reme = divmod(deltatime, 3600)
    minutes, seconds = divmod(rem, 60)
    minutese, secondse = divmod(reme, 60)
    tm = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours),int(minutes),seconds)
    tme = "{:0>2}:{:0>2}:{:02.0f}".format(int(hourse),int(minutese),secondse)
    return tm,tme


def decide_onebyone((clf,data)):
    return [classify_libsvm(clf.wiznet_[cl],clf.map_,clf.nrams_,clf.nobits,clf.notics,clf.nfeatures_,data,clf.ranges_,clf.offsets_) for cl in clf.classes_]

def decide_onebyone_noscale((clf,data)):
    return [classify_libsvm_noscale(clf.wiznet_[cl],clf.map_,clf.nrams_,clf.nobits,clf.notics,clf.nfeatures_,data) for cl in clf.classes_]

def train_onebyone((clf,X,y)):
    for i,data in enumerate(X):
        train_libsvm(clf.wiznet_[clf.classes_[y[i]]],clf.map_, clf.nrams_,clf.nobits,clf.notics,clf.nfeatures_,data,clf.ranges_,clf.offsets_)

def train_onebyone_noscale((clf,X,y)):
    for i,data in enumerate(X):
        train_libsvm(clf.wiznet_[clf.classes_[y[i]]],clf.map_,clf.nrams_,clf.nobits,clf.notics,clf.nfeatures_,data)

def decide_onebyone_b((clf,data)):
    b = clf.b_def
    confidence = 0.0
    res_disc_list = [response_libsvm(clf.wiznet_[cl],clf.map_,clf.nrams_,clf.nobits,clf.notics,clf.nfeatures_,data,clf.ranges_,clf.offsets_) for cl in clf.classes_]
    res_disc = np.array(res_disc_list)
    result_partial = None
    while confidence < clf.conf_def:
        result_partial = np.sum(res_disc >= b, axis=1)
        confidence = calc_confidence(result_partial)
        b += 1
        if(np.sum(result_partial) == 0):
            result_partial = np.sum(res_disc >= 1, axis=1)
            break
    result_sum = np.sum(result_partial, dtype=np.float32)
    if result_sum==0.0:
        result = np.array(np.sum(res_disc, axis=1))/float(clf.nrams_)
    else:
        result = np.array(result_partial)/result_sum
    return result

def decide_onebyone_b_noscale((clf,data)):
    b = clf.b_def
    confidence = 0.0
    res_disc_list = [response_libsvm_noscale(clf.wiznet_[cl],clf.map_,clf.nrams_,clf.nobits,clf.notics,clf.nfeatures_,data) for cl in clf.classes_]
    res_disc = np.array(res_disc_list)
    result_partial = None
    while confidence < clf.conf_def:
        result_partial = np.sum(res_disc >= b, axis=1)
        confidence = calc_confidence(result_partial)
        b += 1
        if(np.sum(result_partial) == 0):
            result_partial = np.sum(res_disc >= 1, axis=1)
            break
    result_sum = np.sum(result_partial, dtype=np.float32)
    if result_sum==0.0:
        result = np.array(np.sum(res_disc, axis=1))/float(clf.nrams_)
    else:
        result = np.array(result_partial)/result_sum
    return result

def calc_confidence(results):
    # get max value
    max_value = results.max()
    if(max_value == 0):  # if max is null confidence will be 0
        return 0
        
    # if there are two max values, confidence will be 0
    position = np.where(results == max_value)
    if position[0].shape[0]>1:
        return 0
        
    # get second max value
    second_max = results[results < max_value].max()
    if results[results < max_value].size > 0:
        second_max = results[results < max_value].max()
        
    # calculating new confidence value
    c = 1 - float(second_max) / float(max_value)
    return c

# CLASSIFIER API
class WisardClassifier(BaseEstimator, ClassifierMixin):
    """Wisard Classifier."""
    wiznet_ = {}
    sharedArray = {}
    results_ = []
    nrams_ = 0
    nloc_ = 0
    ranges_ = []
    offsets_ = []
    classes_ = []
    rowcounter_ = 0
    progress_ = 0.0
    starttm_ = 0
    def __init__(self,n_bits=8,n_tics=256,mapping='random',debug=False,bleaching=True,default_bleaching=1,confidence_bleaching=0.01,n_jobs=1,random_state=0):
        if (not isinstance(n_bits, int) or n_bits<1 or n_bits>32):
            raise Exception('number of bits must be an integer between 1 and 64')
        if (not isinstance(n_tics, int) or n_tics<1):
            raise Exception('number of bits must be an integer greater than 1')
        if (not isinstance(bleaching, bool)):
            raise Exception('bleaching flag must be a boolean')
        if (not isinstance(debug, bool)):
            raise Exception('debug flag must be a boolean')
        if (not isinstance(default_bleaching, int)) or n_bits<1:
            raise Exception('bleaching downstep must be an integer greater than 1')
        if (not isinstance(mapping, str)) or (not (mapping=='random' or mapping=='linear')):
            raise Exception('mapping must either \"random\" or \"linear\"')
        if (not isinstance(confidence_bleaching, float)) or confidence_bleaching<0 or confidence_bleaching>1:
            raise Exception('bleaching confidence must be a float between 0 and 1')
        if (not isinstance(random_state, int)) or random_state<0:
            raise Exception('random state must be an integer greater than 0')
        if (not isinstance(n_jobs, int)) or n_jobs<-1 or n_jobs==0:
            raise Exception('n_jobs must be an integer greater than 0 (or -1)')
        self.nobits = n_bits
        self.notics = n_tics
        self.mapping = mapping
        self.njobs = n_jobs
        self.scaled = False
        if self.njobs == 1:
            self.parallel = False  # set sequential mode
        else:
            self.parallel = True   # set parallel mode
            if self.njobs == -1:
                self.njobs = mp.cpu_count()  # set no. of processes to no. of cores
        self.bleaching = bleaching
        self.b_def = default_bleaching
        self.conf_def = confidence_bleaching
        self.debug = debug
        self.seed = random_state
        self.nloc_ = mypowers[self.nobits]
        return

    # creates input-neurons mappings lists
    def mkmapping_(self, size):
        pixels = np.arange(size)
        map = np.zeros(size, dtype=np.uint32)
        for i in xrange(size):
            j = i + random.randint(0, size - i - 1)
            temp = pixels[i]
            pixels[i] = pixels[j]
            pixels[j] = temp
            map[i] = pixels[i]
        return map
            
    def train_seq_debug(self, X, y):
        self.progress_ = 0.01
        self.starttm_ = time.time()
        for i,data in enumerate(X):
            train_libsvm(self.wiznet_[self.classes_[y[i]]],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,
                         data,self.ranges_,self.offsets_)
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            if self.debug:
                self.progress_ = printProgressBar('train', tm, tme, color.BLUE, color.RED, i+1,self.progress_,len(X))
        if self.debug:
            sys.stdout.write('\n')
        return self

    def train_seq(self, X, y):
        for i,data in enumerate(X):
            train_libsvm(self.wiznet_[self.classes_[y[i]]],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,data,self.ranges_,self.offsets_)
        return self

    def train_seq_debug_noscale(self, X, y):
        self.progress_ = 0.01
        self.starttm_ = time.time()
        for i,data in enumerate(X):
            train_libsvm_noscale(self.wiznet_[self.classes_[y[i]]],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,
                         data)
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            if self.debug:
                self.progress_ = printProgressBar('train', tm, tme, color.BLUE, color.RED, i+1,self.progress_,len(X))
        if self.debug:
            sys.stdout.write('\n')
        return self

    def train_seq_noscale(self, X, y):
        for i,data in enumerate(X):
            train_libsvm_noscale(self.wiznet_[self.classes_[y[i]]],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,data)
        return self

    def decision_function_par(self,X):      # parallel version (no debug no bleaching)
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[len(X), len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        D = pool.map(decide_onebyone, jobs_args)
        return D

    def decision_function_par_noscale(self,X):      # parallel version (no debug no bleaching)
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[len(X), len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        D = pool.map(decide_onebyone_noscale, jobs_args)
        return D

    def decision_function_seq(self,X):      # sequential version (no debug no bleaching)
        D = np.empty(shape=[len(X), len(self.classes_)])
        for i,data in enumerate(X):
            D[i] = [classify_libsvm(self.wiznet_[cl],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,data,self.ranges_,self.offsets_) for cl in self.classes_]
        return D

    def decision_function_seq_noscale(self,X):      # sequential version (no debug no bleaching)
        D = np.empty(shape=[len(X), len(self.classes_)])
        for i,data in enumerate(X):
            D[i] = [classify_libsvm_noscale(self.wiznet_[cl],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,data) for cl in self.classes_]
        return D

    def decision_function_par_debug(self,X):
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[0, len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        self.starttm_ = time.time()
        self.progress_ = 0.01
        D = pool.map(decide_onebyone, jobs_args)
        tm,tme = compTime(time.time()-self.starttm_,self.progress_)
        self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, len(X),self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_par_debug_noscale(self,X):
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[0, len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        self.starttm_ = time.time()
        self.progress_ = 0.01
        D = pool.map(decide_onebyone_noscale, jobs_args)
        tm,tme = compTime(time.time()-self.starttm_,self.progress_)
        self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, len(X),self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_seq_debug(self,X):
        D = np.empty(shape=[0, len(self.classes_)])
        cnt = 0
        self.starttm_ = time.time()
        self.progress_ = 0.01
        for data in X:
            res = [classify_libsvm(self.wiznet_[cl],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,
                                   data,self.ranges_,self.offsets_) for cl in self.classes_]
            D = np.append(D, [res],axis=0)
            cnt += 1
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, cnt,self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_seq_debug_noscale(self,X):
        D = np.empty(shape=[0, len(self.classes_)])
        cnt = 0
        self.starttm_ = time.time()
        self.progress_ = 0.01
        for data in X:
            res = [classify_libsvm_noscale(self.wiznet_[cl],self.map_,self.nrams_,self.nobits,self.notics,self.nfeatures_,data) for cl in self.classes_]
            D = np.append(D, [res],axis=0)
            cnt += 1
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, cnt,self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_par_b(self,X):    # parallel version (no debug with bleaching)
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[0, len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        D = pool.map(decide_onebyone_b, jobs_args)
        return D

    def decision_function_par_b_noscale(self,X):    # parallel version (no debug with bleaching)
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[0, len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        D = pool.map(decide_onebyone_b_noscale, jobs_args)
        return D

    def decision_function_seq_b(self,X):    # sequential version (no debug with bleaching)
        D = np.empty(shape=[0, len(self.classes_)])
        for data in X:
            res = decide_onebyone_b((self,data))  # classify with bleaching (Work in progress)
            D = np.append(D, [res],axis=0)
        return D

    def decision_function_par_b_debug(self,X):
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[0, len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        self.starttm_ = time.time()
        self.progress_ = 0.01
        D = pool.map(decide_onebyone_b, jobs_args)
        tm,tme = compTime(time.time()-self.starttm_,self.progress_)
        self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, len(X),self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_par_b_debug_noscale(self,X):
        pool = mp.Pool(processes=self.njobs)
        D = np.empty(shape=[0, len(self.classes_)])
        jobs_args = [(self,data) for data in X]
        self.starttm_ = time.time()
        self.progress_ = 0.01
        D = pool.map(decide_onebyone_b_noscale, jobs_args)
        tm,tme = compTime(time.time()-self.starttm_,self.progress_)
        self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, len(X),self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_seq_b_debug(self,X):
        D = np.empty(shape=[0, len(self.classes_)])
        cnt = 0
        self.starttm_ = time.time()
        self.progress_ = 0.01
        for data in X:
            res = decide_onebyone_b((self,data))  # classify with bleaching (Work in progress)
            D = np.append(D, [res],axis=0)
            cnt += 1
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, cnt,self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_seq_b_debug_noscale(self,X):
        D = np.empty(shape=[0, len(self.classes_)])
        cnt = 0
        self.starttm_ = time.time()
        self.progress_ = 0.01
        for data in X:
            res = decide_onebyone_b_noscale((self,data))  # classify with bleaching (Work in progress)
            D = np.append(D, [res],axis=0)
            cnt += 1
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, cnt,self.progress_,len(X))
        sys.stdout.write('\n')
        return D

    def decision_function_(self,X):
        None

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.nclasses_ = len(self.classes_)
        self.size_, self.nfeatures_ = X.shape
        self.npixels_ = self.notics * self.nfeatures_
        if self.npixels_ % self.nobits == 0:
            self.nrams_ = self.npixels_ / self.nobits;
        else:
            self.nrams_ = self.npixels_ / self.nobits + 1
        self.map_ = self.mkmapping_(self.npixels_)
        for cl in self.classes_:
            self.wiznet_[cl] = np.zeros(shape=[self.nrams_,self.nloc_], dtype=np.float32)

        self.ranges_ = X.max(axis=0)-X.min(axis=0)
        self.offsets_ = X.min(axis=0)
        self.ranges_[self.ranges_ == 0] = 1

        if np.sum(self.offsets_ != 0) == 0 and np.sum(self.ranges_ != 1) == 0:   # if dataset is already scaled in (0.0,1.0)
            self.scaled = True
        if self.scaled:
            if self.parallel:
                if self.debug:
                    return self.train_seq_debug_noscale(X, y)
                else:
                    return self.train_seq_noscale(X, y)
            else:
                if self.debug:
                    return self.train_seq_debug_noscale(X, y)
                else:
                    return self.train_seq_noscale(X, y)
        else:
            if self.parallel:
                if self.debug:
                    return self.train_seq_debug(X, y)
                else:
                    return self.train_seq(X, y)
            else:
                if self.debug:
                    return self.train_seq_debug(X, y)
                else:
                    return self.train_seq(X, y)
    
    def predict(self, X):
        if self.scaled:
            if self.parallel:
                if self.debug:
                    if self.bleaching:
                        D = self.decision_function_par_b_debug_noscale(X)
                    else:
                        D = self.decision_function_par_debug_noscale(X)
                else:
                    if self.bleaching:
                        D = self.decision_function_par_b_noscale(X)
                    else:
                        D = self.decision_function_par_noscale(X)
            else:
                if self.debug:
                    if self.bleaching:
                        D = self.decision_function_seq_b_debug_noscale(X)
                    else:
                        D = self.decision_function_seq_debug_noscale(X)
                else:
                    if self.bleaching:
                        D = self.decision_function_seq_b_noscale(X)
                    else:
                        D = self.decision_function_seq_noscale(X)
        else:
            if self.parallel:
                if self.debug:
                    if self.bleaching:
                        D = self.decision_function_par_b_debug(X)
                    else:
                        D = self.decision_function_par_debug(X)
                else:
                    if self.bleaching:
                        D = self.decision_function_par_b(X)
                    else:
                        D = self.decision_function_par(X)
            else:
                if self.debug:
                    if self.bleaching:
                        D = self.decision_function_seq_b_debug(X)
                    else:
                        D = self.decision_function_seq_debug(X)
                else:
                    if self.bleaching:
                        D = self.decision_function_seq_b(X)
                    else:
                        D = self.decision_function_seq(X)
        return self.classes_[np.argmax(D, axis=1)]
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"n_bits": self.nobits, "n_tics": self.notics, "mapping": self.mapping, "debug": self.debug, "bleaching": self.bleaching,
            "default_bleaching": self.b_def, "confidence_bleaching": self.conf_def, "random_state": self.seed, "n_jobs": self.njobs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def print_discr(self, cl):
        printDiscr(self.wiznet_[cl])
    
