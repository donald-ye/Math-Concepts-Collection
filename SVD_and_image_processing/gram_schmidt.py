"""
The Singular Value Decomposition (SVD) has many applications in pure mathematics, applied mathematics, and data science. A common theme of many applications of SVD is that for a matrix $A$, by using SVD, we can find a new matrix $A_k$ which is a good approximation of $A$, but the rank of $A_k$ is at most $k$. In general, a small rank matrix can be described with a lower number of entries; we can regard $A_k$ as a "compression" of $A$.

The goal of this project is twofold. First of all, we investigate how to compress image data using the already implemented SVD calculation method. Secondly, we will make a code for a few steps of the SVD calculation.

1. (10 pts) Construct a method **GramSchmidt(A)** where $A = [\mathbf{v}_1, \mathbf{v}_2, \cdots, \mathbf{v}_n]$ is an invertible matrix, and its output is an orthogonal matrix $Q = [\mathbf{w}_1, \mathbf{w}_2, \cdots, \mathbf{w}_n]$ where its column vectors is an orthonormal basis obtained by applying the Gram-Schmidt process to $\{\mathbf{v}_1, \mathbf{v}_2, \cdots, \mathbf{v}_n\}$.
"""

import numpy as np

def GramSchmidt(A):
  n = A.shape[1] #.shape num of columns
  Q = np.zeros_like(A, dtype=float)

  # A[:, i] returns column i. A[i, :] return row
  for i in range(n):
    qi = A[:, i]
    for j in range(i):
      qj = Q[:, j]
      qi -= np.dot(qj, A[:, i]) * qj # subtracts projection
    Q[:, i] = qi / np.linalg.norm(qi) # normalize and store it in i-th column of Q
  return Q

A = np.array([[1, 1],
              [1, 0]],dtype=float)
GramSchmidt(A)

T = np.array([[1.,2.,3.],[4.,5.,6.],[2.,2.,1.]])
print(GramSchmidt(T))
# output of A should be

# Q = np.array([0.70710678  0.70710678],
#              [0.70710678 -0.70710678]])
