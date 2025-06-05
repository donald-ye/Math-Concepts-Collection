""" 2. (10 pts) Create a function **FourierCoeff(f, n)** which returns the list of Fourier coefficients $a_0, a_1, \cdots, a_n$ and $b_1, b_2, \cdots, b_n$ where $a_0 = \langle f(x), 1\rangle$, $a_k = \langle f(x), \cos kx\rangle$ for $k \ge 1$, $b_k = \langle f(x), \sin kx\rangle$. To calculate each coefficient, use the function Simpson(f, m) with $m = 200$."""

import numpy as np

def FourierCoeff(f, n):
  list_a = [] # a coefficients
  list_b = [] # b coefficients

  a0 = (1/np.pi) * Simpson(f, 200)
  list_a.append(a0)

  for i in range(1, n+1):
    f_cos_ks = lambda x: f(x) * np.cos(i * x)
    f_sin_ks = lambda x: f(x) * np.sin(i * x)

    a_k = (1/np.pi) * Simpson(f_cos_ks, 200)
    b_k = (1/np.pi) * Simpson(f_sin_ks, 200)

    list_a.append(a_k)
    list_b.append(b_k)

  return list_a, list_b

n = int(input("How many coefficients would you like?"))
# NOTE: given the conditions: (len(range(list_a)) == len(range(list_b)) + 1)

list_a, list_b = FourierCoeff(f, n)
list_a = list(map(float, list_a)) # convert from numpy float -> python float
list_b = list(map(float, list_b)) # Makes ouput look cleaner
print("a_k coefficients:", list_a)
print("b_k coefficients:", list_b)

""" 
3. (10 pts) Create a function **DiscreteFourierCoeff(f, n)** which returns the list of Fourier coefficients $a_0, a_1, \cdots, a_n$ and $b_1, b_2, \cdots, b_n$. We use $m = 100$. Here $a_0 = \langle \mathbf{y}, \phi_0\rangle$, $a_k = \langle \mathbf{y}, \phi_k\rangle$ for $k \ge 1$, $b_k = \langle \mathbf{y}, \psi_k\rangle$. Check the lecture notes for the definition of $\mathbf{y}$, $\phi_k$, and $\psi_k$.
    Note that the initial data is a continuous function, but the approximation is recorded as a list of $2n+1$ numbers!
"""

def DiscreteFourierCoeff(f, n):
  m = 100
  xs = np.linspace(-np.pi, np.pi, m, endpoint=False)
  ys = np.array([f(x) for x in xs])
  list_a = [] # a coefficients
  list_b = [] # b coefficients

  a0 = (1/m) * np.sum(ys) * 2
  list_a.append(a0)

  for i in range(1, n+1):
    phi_k = np.cos(i * xs)
    psi_k = np.sin(i * xs)

    a_k = (2/m) * np.dot(ys, phi_k)
    b_k = (2/m) * np.dot(ys, psi_k)

    list_a.append(a_k)
    list_b.append(b_k)

  return list_a, list_b

n = int(input("How many coefficients would you like?"))
# NOTE: given the conditions: (len(range(list_a)) == len(range(list_b)) + 1)

list_a, list_b = DiscreteFourierCoeff(f, n)
list_a = list(map(float, list_a)) # convert from numpy float -> python float
list_b = list(map(float, list_b)) # Makes ouput look cleaner
print("a_k coefficients:", list_a)
print("b_k coefficients:", list_b)
