# 3. (15 pts) Create a Python function **secant(x0, x1)** that finds the root of $N(t) - 1 = 0$ by secant method. $p_0 = x0$ and $p_1 = x1$.

# <ul>
#     <li>Use an error bound $10^{-6}$. You may estimate the error size by $|p_{n} - p_{n-1}|$.</li>
#     <li>Allow at most 1000 iterations.</li>
#     <li>For each step, print $p_n$ and the estimation of an error $|p_n - p_{n-1}|$.</li>
# </ul>

import math

def secant(x0, x1) :
  # Setting up initial values
  p_n0 = x0
  p_n1 = x1
  p_ni = p_n1 - ((N(p_n1) - 1) * (p_n1 - p_n0)/ ((N(p_n1) - 1) - (N(p_n0) - 1)))
  error = abs(p_ni - p_n1)
  i = 2

  print ("p_0 = " + str(p_n0))
  print ("p_1 = " + str(p_n1))

  while error > 10**(-6) :
    p_ni = p_n1 - ((N(p_n1) - 1) * (p_n1 - p_n0)/ ((N(p_n1) - 1) - (N(p_n0) - 1)))
    error = abs(p_ni - p_n1)

    # Printing p_n and the estimation of the error
    print ("p_" + str(i) + " = " + str(p_ni) + ", error = " + str(error))

    # Assigns next values
    p_n0 = p_n1
    p_n1 = p_ni
    i = i + 1

# TEST
secant(10,20)
