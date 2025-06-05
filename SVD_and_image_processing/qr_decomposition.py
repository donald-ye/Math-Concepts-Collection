"""2. Construct a method **QRdecomposition(A)** where $A$ is an invertible matrix and the output is a pair of matrices $[Q, R]$, that is the QR decomposition of $A$. In other words, $A = QR$ and $Q$ is an orthogonal matrix and $R$ is an upper triangular matrix."""

def QRdecomposition(A): #process is very similar to example above
  n = A.shape[1]
  Q = np.zeros_like(A, dtype=float)
  R = np.zeros((n, n), dtype=float) # upper traiangular

  for i in range(n):
    qi = A[:, i]
    for j in range(i):
      qj = Q[:, j]
      R[j, i] = np.dot(qj, A[:, i])
      qi -= R[j, i] * qj
    R[i, i] = np.linalg.norm(qi)
    Q[:, i] = qi / R[i,i]
  return Q, R

A = np.array([[1, 1],
              [1, 0]],dtype=float)
Q, R = QRdecomposition(A)

print("Orthogonal Matrix Q = \n", Q)
print("Upper Triangular Matrix R = \n", R)

print(QRdecomposition(T))
