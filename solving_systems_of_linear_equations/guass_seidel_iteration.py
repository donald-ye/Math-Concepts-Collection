"""5. Create a function **GaussSeidel(A, b, err)** which solves a system of linear equations $Ax = b$ by using Gauss-Seidel interation method. Set $x^{(0)} = \vec{0}$. We stop the iteration when the estimation of the error $||x^{(k)} - x^{(k-1)}||_\infty$ is less than err or $k = 1000$. (Here $x^{(k)}$ is the $k$-th output of the iteration)."""

def GaussSeidel(A, b, err):
  n = len(A)
  A = A.astype(float)
  b = b.astype(float)

  # Check if A is Diagonally_dominant # utilizes previous function
  if not Diagonally_dom(A):
    print("Non-Diagonally Dominant Matrix detected. Gauss-Seidel Iteration may not converge.")
    return None

  x = np.zeros(n)
  D_inv = 1 / np.diag(A)

  k = 1000  #iterations
  for i in range(k):
    x_old = x.copy()
    for j in range(n):
      sum1 = np.dot(A[j, :j], x[:j])
      sum2 = np.dot(A[j, j+1:], x[j+1:])
      x[j] = D_inv[j] * (b[j] - sum1 - sum2)

    if np.max(np.abs(x - x_old)) < err:
      return x

  print(f"Gauss-Seidel method did not converge within {k} iterations.")
  return x

#requires a Strictly Diagonally Dominant Matrix

A = np.array([[4, 1, -1],
              [3, 6, 2],
              [1, 1, 5]], dtype=float)
b = np.array([4, 10, 3], dtype=float)
err = 1e-8 # Arbitrary
x = GaussSeidel(A, b, err)

print("In the format Ax=b, given known values of \nA= \n", A ,"and \nb= \n",b ,"\nUtilizing the Gauss Seide Iteration Method, we conclude that\nx =\n ", x)

# # Non-Diagonally Dominant Matrix test case
# A = np.array([[2, 1, -1],
#               [3, 2, 2],
#               [1, 1, 1]], dtype=float)
# b = np.array([4, 10, 3], dtype=float)
# err = 1e-8
# print(Jacobi(A, b, err))

print(GaussSeidel(tA,tb,0.00001))
