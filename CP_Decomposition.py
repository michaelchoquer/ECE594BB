import numpy as np
from scipy.linalg import khatri_rao
import tensorly as ts

#TODO implement in a function

# start off with an example
chi = (np.arange(24) + 1).reshape(2, 3, 4)

r_guess = 3
a_guesses = np.array([np.random.random((np.size(chi, n), r_guess)) for n in np.arange(np.ndim(chi))], dtype=object)
# print(np.shape(a_guesses[0]))

# loop until happy with the result
# set up a while loop here with an error condition
v_total = np.eye(r_guess)
for m in np.arange(np.ndim(chi)):

    for n in np.arange(np.ndim(chi)):
        # do a cascaded multiplication of all the factor matrices
        v_total = np.multiply(v_total, np.matmul(a_guesses[n].T, a_guesses[n]))

    # intermediate result: get the khatri-rao product of all the factor matrices
    a_khatri_prod = a_guesses[np.ndim(chi) - 1] # start with last matrix
    for p in np.arange(np.ndim(chi), -1, -1):
        a_khatri_prod = khatri_rao(a_khatri_prod, a_guesses[p])

    a_guesses[n] = np.matmul(np.matmul(ts.unfold(chi, n), a_khatri_prod), v_total.T)
    # figure out a way to normalize a_guesses

