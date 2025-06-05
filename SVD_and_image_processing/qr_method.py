"""##### 3. (10 pts) A key step on the diagonalization of a symmetric matrix (and hence on SVD) is the QR method. Construct a method **QRalgorithm(A, err)** where $A$ is a symmetric tridiagonal matrix, err is a positive real number, and output is a list of eigenvalues of $A$. Let $A^{(k)}$ be the output of $k$-th iteration (See the notation in the lecture note) and let $e^{(k)}$ be the vector consisting of diagonal entries of $A^{(k)}$. Stop the iteration if either


*   $k = 1000$ or;
*   $||e^{(k)} - e^{(k-1)}||_{\infty} < \mathrm{err}$.

##### Let $M$ be a $(10 \times 10)$ symmetric tridiagonal matrix such that $$M_{ij} = \begin{cases}11-i, & \mbox{if } i = j,\\1, & \mbox{if } i = j+1 \mbox{ or } i = j-1,\\0, & \mbox{otherwise}.\end{cases}$$
##### By using **QRalgorithm(A, err)**, compute eigenvalues of $M$. Set $\mathrm{err} = 10^{-5}$.
"""

def generate_symmetric_tridiagonal_matrix(n): #creation of M
  M = np.zeros((n, n))
  for i in range(n):
    M[i, i] = 11 - i
    if i > 0:
      M[i, i - 1] = 1
    if i < n - 1:
      M[i, i + 1] = 1
  return M


def QRalgorithm(A, err):
  n = A.shape[0] #size of rows
  eig_old = np.diagonal(A)
  max_iterations = 1000

  for i in range(max_iterations):
    Q, R = QRdecomposition(A)
    A = np.dot(R, Q) #update A
    eig_new = np.diagonal(A)
    if np.linalg.norm(eig_new - eig_old, np.inf) < err: # break condition
      break

    eig_old = eig_new #update old eigenvalues
  return eig_new

M = generate_symmetric_tridiagonal_matrix(10)

print(M)
eigenvalues = QRalgorithm(M, 1e-5)

print("Eigenvalues in the matrix: \n", eigenvalues)

"""From now on, we will discuss image processing with Python. For simplicity, we are going to use a grayscale (black and white) image only. Below is how to convert a grayscale image to a python matrix. Matplotlib can only read the .png file natively."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import numpy.linalg as la

response = requests.get('https://drive.google.com/uc?export=view&id=1Di59ub7nRPQRWpWZqaxWUzBRGnny8imU', stream=True)
sloth_img = Image.open(BytesIO(response.content))
sloth_img_array = np.array(sloth_img)[:,:,0]
# These three lines read 'sloth_gray.png' and record it as an array.

plt.imshow(sloth_img_array, cmap='gray')
# A matrix can be converted and shown to a grayscale image.

"""In the above code, <b>sloth_img_array</b> is a matrix. Thus we can compute its SVD and use it to do some image processing.

For a matrix $A$, <b>svd</b> method in the linear algebra module can be used to calculate its SVD. The output is a triplet of data $U$, $D$, and $V^t$, where $U$ and $V^t$ are orthogonal matrices and $D$ is a list of singular values of $A$ (not a diagonal matrix!). So if we denote $S$ as the diagonal matrix whose diagonal entries are numbers on $D$, then $A = USV^t$.
"""

A = np.array([[1,0,1],[0,1,0],[0,1,1],[0,1,0],[1,1,0]])
U, D, Vt = la.svd(A, full_matrices = True)
print("U=", U)
print("D=", D)
print("V^t=", Vt)
