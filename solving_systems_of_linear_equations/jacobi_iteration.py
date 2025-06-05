"""4. Create a function **Jacobi(A, b, err)** which solves a system of linear equations $Ax = b$ by using Jacobi interation method. Set $x^{(0)} = \vec{0}$. We stop the iteration when the estimation of the error $||x^{(k)} - x^{(k-1)}||_\infty$ is less than err or $k = 1000$. (Here $x^{(k)}$ is the $k$-th output of the iteration).

"""

def Diagonally_dom(A):
  n = len(A)
  for i in range(n):
    if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
      return False
  return True

def Jacobi(A, b, err):
  n = len(A)
  A = A.astype(float)
  b = b.astype(float)

  # Check if A is Diagonally_dominant
  if not Diagonally_dom(A):
    print("Non-Diagonally Dominant Matrix detected. Jacobi Iteration may not converge.")
    return None

  x_old = np.zeros(n)
  D_inv = 1 / np.diag(A)
  R = A - np.diag(np.diag(A))

  k = 1000  # iterations
  for i in range(k):
    x_new = D_inv * (b - np.dot(R, x_old))

    if i % 10 == 0:
      if np.linalg.norm(x_new - x_old, np.inf) < err:
        return x_new

    x_old[:] = x_new

  print(f"Jacobi method did not converge within {k} iterations.")
  return x_new

 #requires a Strictly Diagonally Dominant Matrix
A = np.array([[4, 1, -1],
              [3, 6, 2],
              [1, 1, 5]], dtype=float)
b = np.array([4, 10, 3], dtype=float)
err = 1e-8 # Arbitrary
x = Jacobi(A, b, err)

print("In the format Ax=b, given known values of \nA= \n", A ,"and \nb= \n",b ,"\nUtilizing the Jacobi Iteration Method, we conclude that\nx =\n ", x)

print(Jacobi(tA,tb,0.00001))
# # Non-Diagonally Dominant Matrix test case
# A = np.array([[2, 1, -1],
#               [3, 2, 2],
#               [1, 1, 1]], dtype=float)
# b = np.array([4, 10, 3], dtype=float)
# err = 1e-8
# print(Jacobi(A, b, err))
