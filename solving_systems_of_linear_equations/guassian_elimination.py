"""

For this project, <b>DO NOT</b> use the 'solve' or 'inv' method in the linear algebra package! You have to create your own code for it.

1. (5 pts) Create a function **randmat(n)** which returns a random square matrix constructed as the following recipe.
<ul>
    <li>The size of the matrix is $n \times n$.</li>
    <li>Each off-diagonal entry ($a_{ij}$ where $i \ne j$) is a random number in $[0, 1)$. A random number can be constructed by the random method (see <a href="https://docs.scipy.org/doc/numpy/reference/routines.random.html">here</a>).</li>
    <li>A diagonal entry $a_{ii}$ is a random number in $[n, n+1)$. (This condition guarantees that the matrix $(a_{ij})$ is strictly diagonally dominant, hence invertible.)</li>
</ul>

And create a function **randvec(n)** which returns an $n$-dimensional random vector whose entries are random numbers in $[0, 100)$.
"""

import numpy as np

def randvec(n):
  return np.random.rand(n) * 100

def randmat(n):
  matrix = np.random.random((n,n))
  np.fill_diagonal(matrix, np.random.uniform(n, n+1, size=n))
  for i in range(n):
    for j in range(n):
        if i != j:
          matrix[i][j] = np.random.random()

  return matrix

n = int(input("Enter the n-dimensional size of your matrix: "))

tA = randmat(n)
tb = randvec(n)
#print("Random matrix: \n",randmat(n)) # Diagonals = (n, n+1), non-Diagonals = random[0,1)
#print("Random vector: \n",randvec(n))
print("Random matrix: \n",tA) # Diagonals = (n, n+1), non-Diagonals = random[0,1)
print("Random vector: \n",tb)

"""2. (10 pts) Create a function **GaussElim(A, b)** which solves a system of linear equations $Ax = b$ by using Gaussian Elimination with the partial pivoting."""

def GaussElim(A,b):
  n = len(b)
  A = A.astype(float)
  b = b.astype(float)

  aug_matrix = np.hstack((A, b.reshape(n, 1)))

  for i in range(n):
    max_row = np.argmax(np.abs(aug_matrix[i:, i])) + i

    if i != max_row:
      aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]

    # Checks if pivot is zero, executes faster than (np.linalg.det(A) = 0)
    if np.isclose(aug_matrix[i, i], 0):
      print("Singular matrix detected. No unique solution exists.")
      return None

    aug_matrix[i] = aug_matrix[i] / aug_matrix[i, i]
    for j in range(i + 1, n):
      aug_matrix[j] -= aug_matrix[j, i] * aug_matrix[i]

  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    x[i] = aug_matrix[i, -1] - np.dot(aug_matrix[i, i+1:n], x[i+1:])

  return x

A = np.array([[2, 1, -1],
              [3, 2, 2],
              [1, 1, 1]], dtype=float)

b = np.array([4, 10, 3], dtype=float)
x = GaussElim(A,b)

print("Solution of vector x:", x)

print(GaussElim(tA,tb))
# # Singular Matrix test case

# A = np.array([[2, 1, -1],
#               [4, 2, -2],
#               [1, 1, 1]], dtype=float)  # Second row is 2 * first row

# b = np.array([4, 8, 3], dtype=float)
# print(GaussElim(A, b))
