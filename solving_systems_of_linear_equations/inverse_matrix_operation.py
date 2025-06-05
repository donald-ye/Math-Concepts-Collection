"""3. Create a function **InvMat(A,b)** which solves a system of linear equations $Ax = b$ with the "theoretically simplest method," that is, computing $x = A^{-1}b$. Compute the inverse matrix as the following:
<ul>
    <li>Make an augmented matrix $[A | I]$ where $I$ is the $n \times n$ identity matrix.</li>
    <li>Apply elementary row operations until the left half $A$ on $[A| I]$ becomes $I$, so it looks $[I | B]$.</li>
    <li>Then the right half of the augmented matrix $B$ is $A^{-1}$.</li>
</ul>
"""

def InvMat(A,b):
  n = len(A)
  A = A.astype(float)
  b = b.astype(float)

  aug_matrix = np.hstack((A, np.identity(n)))

  # Check if A is square
  if A.shape[0] != A.shape[1]:
      print("Non-Square matrix detected, matrix can not be inverted.")
      return None

  for i in range(n):
    aug_matrix[i] = aug_matrix[i] / aug_matrix[i, i]
    for j in range(i+1, n):
        aug_matrix[j] -= aug_matrix[j, i] * aug_matrix[i]

  for i in range(n-1, -1, -1):
    for j in range(i-1, -1, -1):
      aug_matrix[j] -= aug_matrix[j, i] * aug_matrix[i]

  # Inverse and solution
  A_inv = aug_matrix[:, n:]
  x = np.dot(A_inv, b)
  return x

A = np.array([[2, 1, -1],
              [3, 2, 2],
              [1, 1, 1]], dtype=float)
b = np.array([4, 10, 3], dtype=float)
x = InvMat(A,b)

print("In the format Ax=b, given known values of \nA= \n", A ,"and \nb= \n",b ,"\nUtilizing the inverse matrix, we conclude that\nx =\n ", x)

# # Non-Square matrix test case
# A = np.array([
#     [2, 1],
#     [4, 2],
#     [1, 3]
# ], dtype=float)

# b = np.array([5, 10, 6], dtype=float)
# print(InvMat(A,b))

print(InvMat(tA,tb))
