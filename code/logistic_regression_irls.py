#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = 0.5

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]
# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

# Initialize w.
w = np.array([0.1, 0, 0])

# Error values over all iterations.
e_all = []

DATA_FIG = 1

# Set up the slope-intercept figure
SI_FIG = 2
plt.figure(SI_FIG)
plt.rcParams.update({'font.size': 15})
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

for iter in range(0, max_iter):

    y = sps.expit(np.dot(X, w))

    e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
    e_all.append(e)

    w_old = w

    Theta = X
    t = t
    y = sps.expit(np.dot(X, w))
    R = np.diag(y * (1 - y))
    z = np.dot(Theta, np.transpose(np.matrix(w))) - np.dot(np.linalg.inv(R), np.transpose(np.matrix(y - t)))

    tmp1 = np.dot(np.dot(np.transpose(Theta), R), Theta)
    tmp2 = np.linalg.inv(tmp1)

    w = np.dot(np.dot(np.dot(tmp2, np.transpose(Theta)), R), z)
    w = np.array(w.A1)

    plt.figure(SI_FIG)
    a2.plot_mb(w, w_old)

    print("epoch {0:d}, negative log-likelihood {1:.4f}, w={2}".format(iter, e, w.T))

    if iter > 0:
        if np.absolute(e - e_all[iter - 1]) < tol:
            break

# Plot error over iterations
plt.figure()
plt.plot(e_all)
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression using iterative reweighted least squares')
plt.xlabel('Epoch')
plt.show()
