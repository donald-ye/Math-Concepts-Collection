"""##### 5. (10 pts) Prepare your favorite image file in a grayscale .png format. (A photo is better than computer graphics. I suggest using a picture smaller than $500 \times 500$.) Load the image file and plot the original image. Convert the image as a matrix $A$. Run **SVDcompression(A,k)** for $k = 1, 5, 10$, and $50$."""

response = requests.get("https://drive.google.com/uc?export=download&id=1IsT5WO2-USQK-6F5UWidqTsKezj8BrUJ", stream=True)
favorite_img = Image.open(BytesIO(response.content))
favorite_img_array = np.array(favorite_img)[:,:,0]

plt.imshow(favorite_img_array, cmap='gray')

#original image in grayscale
plt.imshow(favorite_img_array, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

SVDcompression(favorite_img_array, k=1)
SVDcompression(favorite_img_array, k=5)
SVDcompression(favorite_img_array, k=10)
SVDcompression(favorite_img_array, k=50)

"""##### 6. (30 pts, extra credit) Create a method **SVDcalculation(A)** which computes the singular value decomposition of $A \in M_{m \times n}$ (with $m \ge n$) of a full rank matrix $A$ from scratch. Its output is a triple of matrices $[U, S, V]$ where
* $U$ is an $m \times m$ orthogonal matrix;
* $S$ is an $m \times n$ diagonal matrix with a positive decreasing diagonal entries;
* $V$ is an $n \times n$ orthogonal matrix;
* $A = USV^t$.

For the diagonalization of a symmetric matrix, use **QRalgorithm(A, err)** with $err = 10^{-5}$.

The only missing part of the SVD computation is the Householder reduction, which finds for a given symmetric matrix $A$ a similar tridiagonal matrix $B$. You may use the following command **hessenberg**. It returns two matrices $H$ and $Q$ such that
* $H$ is a tridiagonal matrix;
* $Q$ is an orthogonal matrix;
* $A = QHQ^t$.
"""

from scipy.linalg import hessenberg
A = np.array([[1,2,3,4],[2,5,6,7],[3,6,8,9],[4,7,9,10]])
print(A)
H, Q = hessenberg(A, calc_q=True)
# This command finds two matrices H, Q such that A = QH
print(H)
print(Q)
print(Q@H@(Q.T))

import numpy as np
from scipy.linalg import hessenberg

# simplified version
def QRalgorithm(A, err = 1e-5):
  n = A.shape[0]
  eig_old = np.diagonal(A)
  max_iterations = 1000

  for i in range(max_iterations):
    Q, R = np.linalg.qr(A)
    A = np.dot(R, Q)
    eig_new = np.diagonal(A)
    if np.linalg.norm(eig_new - eig_old, np.inf) < err:
        break
    eig_old = eig_new
  return eig_new, Q

def SVDcalculation(A):
  m, n = A.shape
  H, Q = hessenberg(A, calc_q=True)  # H is the tridiagonal matrix, Q is orthogonal matrix
  eigenvalues, eigenvectors = QRalgorithm(H)

  singular_values = np.sqrt(np.abs(eigenvalues))
  S = np.zeros((m, n))
  np.fill_diagonal(S, singular_values)

  # U is obtained by multiplying Q from the Householder reduction with the eigenvectors of H
  U = np.dot(Q, eigenvectors)
  V = eigenvectors

  return U, S, V

A = np.array([[1, 2, 3],
              [2 ,5 ,6],
              [3 ,6 ,8],
              [4 ,7 ,9]])

# SVD computation
U, S, V = SVDcalculation(A)
print("A matrix:")
print(A)
print("\nU matrix:")
print(U)
print("\nS matrix:")
print(S)
print("\nV matrix:")
print(V)

# Verification that A = USV^t
A_reconstructed = np.dot(U, np.dot(S, V.T))
print("\nReconstructed A:")
print(A_reconstructed)

# Answers: I'm getting cooked
# U matrix:
# [[-0.20739292  0.34827934 -0.92797369 -0.02220475]
#  [-0.46331061  0.70750698  0.4677419   0.14114043]
#  [-0.7192283   0.27869298  0.06571004 -0.64177527]
#  [-0.975146   -0.37112085 -0.29930304  0.1843296 ]]

# S matrix:
# [[17.99913415  0.          0.          0.        ]
#  [ 0.          1.25025102  0.          0.        ]
#  [ 0.          0.          0.13208361  0.        ]
#  [ 0.          0.          0.          0.01687902]]

# V matrix:
# [[-0.20739292 -0.46331061 -0.7192283  -0.975146  ]
#  [ 0.34827934  0.70750698  0.27869298 -0.37112085]
#  [-0.92797369  0.4677419   0.06571004 -0.29930304]
#  [-0.02220475  0.14114043 -0.64177527  0.1843296 ]]

# Reconstructed A:
# [[ 1.00000000e+00  2.00000000e+00  3.00000000e+00  4.00000000e+00]
#  [ 2.00000000e+00  5.00000000e+00  6.00000000e+00  7.00000000e+00]
#  [ 3.00000000e+00  6.00000000e+00  8.00000000e+00  9.00000000e+00]
#  [ 4.00000000e+00  7.00000000e+00  9.00000000e+00  1.00000000e+01]]
