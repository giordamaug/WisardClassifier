
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
    print(tuple)
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


