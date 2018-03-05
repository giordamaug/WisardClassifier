"""
    WiSARD C Library Wrapper in Python
    
    Created by Maurizio Giordano on 13/12/2016
    
"""

from ctypes import *
import numpy as np
import os
import random
import platform
if platform.system() == 'Linux':
    suffix = '.so'
elif platform.system() == 'Windows':
    suffix = '.dll'
elif platform.system() == 'Darwin':
    suffix = '.dylib'
else:
    raise Error("Unsupported Platform")

libpath = "wislib/libwisard-cxx_static_3.0" + suffix
wizl = CDLL(os.path.join(os.environ['PWD'], libpath))

""" Mapping data structure """

c_value = c_float
c_key = c_ulong
_doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
_valuepp = np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
_datapp = np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
_indexpp = np.ctypeslib.ndpointer(dtype=np.uint32,ndim=1,flags='C_CONTIGUOUS')
_keypp = np.ctypeslib.ndpointer(dtype=np.uint64,ndim=1,flags='C_CONTIGUOUS')

""" Discriminator structure """
class Discr(Structure):
    _fields_ = [("n_ram", c_int),
                ("n_bit", c_int),
                ("n_loc", c_ulong),
                ("size", c_int),
                ("tcounter", c_ulong),
                ("rams", _doublepp),
                ("map", POINTER(c_int)),
                ("rmap", POINTER(c_int)),
                ("mi", POINTER(c_value))]

""" Constructor interface """
_make_discr =  wizl.make_discr
_make_discr.restype =  POINTER(Discr)
_make_discr.argtypes = [ c_int, c_int, c_char_p, c_int ]

def make_discr(nbit, size, maptype='random', seed=0):
    return _make_discr(c_int(nbit), c_int(size), c_char_p(maptype), c_int(seed))

""" Train/Classify wrappers"""
_print_discr = wizl.print_discr
_train_discr = wizl.train_discr
_train_discr.argtypes = [ POINTER(Discr), _keypp ]
_classify_discr = wizl.classify_discr
_classify_discr.argtypes = [ POINTER(Discr), _keypp ]
_classify_discr.restype = c_double
_response_discr = wizl.response_discr
_response_discr.argtypes = [ POINTER(Discr), _keypp ]
_response_discr.restype = POINTER(c_double)

_train_tuple = wizl.train_tuple
_train_tuple.argtypes = [ _doublepp, c_int, _keypp ]
_classify_tuple = wizl.classify_tuple
_classify_tuple.argtypes = [ _doublepp, c_int, _keypp ]
_classify_tuple.restype = c_double
_response_tuple = wizl.response_tuple
_response_tuple.argtypes = [ _doublepp, c_int, _keypp]
_response_tuple.restype = POINTER(c_double)

_train_libsvm = wizl.train_libsvm
_train_libsvm.argtypes = [ _doublepp, _indexpp, c_int, c_int, c_int, c_int, _datapp, _datapp, _datapp ]
_classify_libsvm = wizl.classify_libsvm
_classify_libsvm.argtypes = [ _doublepp, _indexpp, c_int, c_int, c_int, c_int, _datapp, _datapp, _datapp ]
_classify_libsvm.restype = c_double
_response_libsvm = wizl.response_libsvm
_response_libsvm.argtypes = [ _doublepp, _indexpp, c_int, c_int, c_int, c_int, _datapp, _datapp, _datapp]
_response_libsvm.restype = POINTER(c_double)

def train_discr(discr, intuple):
    _train_discr(discr, intuple)

def print_discr(discr):
    _print_discr(discr)

def classify_discr(discr, intuple):
    return _classify_discr(discr, intuple)

def response_discr(discr, intuple):
    rawres = _response_discr(discr, intuple)
    return [ rawres[i] for i in range(discr.contents.n_ram)]

def train_tuple(rams, n_ram, intuple):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    _train_tuple(xpp, n_ram, intuple)

def classify_tuple(rams, n_ram, intuple):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    return _classify_tuple(xpp, n_ram, intuple)

def response_tuple(rams, n_ram, intuple):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    rawres = _response_tuple(xpp, n_ram, intuple)
    return [ rawres[i] for i in range(n_ram)]

def train_libsvm(rams, map, n_ram, n_bit, n_tics, n_feature, data, den, off):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    _train_libsvm(xpp, map, n_ram, n_bit, n_tics, n_feature, data, den, off)

def classify_libsvm(rams, map, n_ram, n_bit, n_tics, n_feature, data, den, off):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    return _classify_libsvm(xpp, map, n_ram, n_bit, n_tics, n_feature, data, den, off)

def response_libsvm(rams, map, n_ram, n_bit, n_tics, n_feature, data, den, off):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    rawres = _response_libsvm(xpp, map, n_ram, n_bit, n_tics, n_feature, data, den, off)
    return [ rawres[i] for i in range(n_ram)]

""" Train/Classify wrappers"""
_train_libsvm_noscale = wizl.train_libsvm_noscale
_train_libsvm_noscale.argtypes = [ _doublepp, _indexpp, c_int, c_int, c_int, c_int, _datapp ]
_classify_libsvm_noscale = wizl.classify_libsvm_noscale
_classify_libsvm_noscale.argtypes = [ _doublepp, _indexpp, c_int, c_int, c_int, c_int, _datapp ]
_classify_libsvm_noscale.restype = c_double
_response_libsvm_noscale = wizl.response_libsvm_noscale
_response_libsvm_noscale.argtypes = [ _doublepp, _indexpp, c_int, c_int, c_int, c_int, _datapp ]
_response_libsvm_noscale.restype = POINTER(c_double)

def train_libsvm_noscale(rams, map, n_ram, n_bit, n_tics, n_feature, data):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    _train_libsvm_noscale(xpp, map, n_ram, n_bit, n_tics, n_feature, data)

def classify_libsvm_noscale(rams, map, n_ram, n_bit, n_tics, n_feature, data):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    return _classify_libsvm_noscale(xpp, map, n_ram, n_bit, n_tics, n_feature, data)

def response_libsvm_noscale(rams, map, n_ram, n_bit, n_tics, n_feature, data):
    xpp = (rams.__array_interface__['data'][0] + np.arange(rams.shape[0])*rams.strides[0]).astype(np.uintp)
    rawres = _response_libsvm_noscale(xpp, map, n_ram, n_bit, n_tics, n_feature, data)
    return [ rawres[i] for i in range(n_ram)]


