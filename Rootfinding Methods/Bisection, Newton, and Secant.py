# # Project 1. Rootfinding

# The <b>Gompertz curve</b> or Gompertz function is a type of mathematical model named after Benjamin Gompertz (1779-1865). 
# It is a function that describes growth as being slowest at the start and end of a given time period. Population biology is 
# especially concerned with the Gompertz function. This function is especially useful in describing the rapid growth of a 
# certain population of organisms (such as tumors, bacteria, etc.) while also considering the eventual horizontal asymptote 
# once the carrying capacity is determined. The function was originally designed to describe human mortality, but since has 
# been modified to be applied in biology, with regards to detailing populations.

# It is modeled as follows:

# $$N(t) = N_0 \mathrm{exp}((\ln (N_I/N_0)) (1-\mathrm{exp}(-bt)) = N_0 e^{(\ln \frac{N_I}{N_0}) (1-e^{-bt})}$$

# where $t$ is the time, $N(t)$ is the population at time $t$, $N_0$ is the initial population, $N_I$ is the plateau population 
# number (the maximum capacity in the given situation), $b$ is the initial growth rate, and $exp(x)$ is the exponential function $e^x$. 
# The unit for $N(t)$, $N_I$, and $N_0$ are millions, and the unit for $t$ is hours.

# In this project, we are going to write computer programs that determine the amount of time that it takes for $N(t)$ to rise from 
# the initial population $N_0 = 3\cdot 10^{-5}$ to $1$. We use $N_I = 10^3$ and $b = 0.12$.

# Note that the solution of $N(t) = 1$ is equivalent to $N(t) - 1 = 0$, so this is a root-finding problem.

#### 1. (20 pts) Create a Python function **bisection(b)** that finds the root of $N(t) - 1 = 0$ by the bisection method. The initial interval is $[0, b]$.

# <ul>
#     <li>Use an error bound $10^{-6}$.</li>
#     <li>Allow at most 1000 iterations.</li>
#     <li>For each step, print the left endpoint $a_n$, the right endpoint $b_n$, and the approximation (= midpoint) $p_n$. </li>
# </ul>

import math

def N(t):
  n_0 = 3*10**(-5)
  n_i = 10**3
  b = 0.12

  if t == 0:
    return (3*10**(-5))
  else:
    return (n_0 * math.exp(math.log(n_i/n_0)*(1 - math.exp(-b*t))))

# TEST -- N(2)

# Attempt 2
def bisection(b):
  # calc the min num of iterations required to be within error (round up)
  n = math.ceil(math.log2((b)/10**(-6)))

  # "Allow at most 1000 iterations."
  if n > 1000 :
    print ("Limit of iterations reached")

  # Setting up initial values
  a_n = 0
  b_n = b

  # TEST -- print (n)

  # Loop that performs and prints the operations
  for i in range(1, (n+1)) :
    # Calculating the values of p_n, f_a_n, f_p_n, f_p_n)
    p_n = a_n + ((b_n - a_n)/2)
    f_a_n = N(a_n) - 1
    f_p_n = N(p_n) - 1
    f_b_n = N(b_n) - 1

    # Printing the left endpoint a_n, the right endpoint b_n, and p_n.
    print ("a_" + str(i) + " = " + str(a_n) + ", f(a_" + str(i) + ") = " + str(f_a_n))
    print ("p_" + str(i) + " = " + str(p_n) + ", f(p_" + str(i) + ") = " + str(f_p_n))
    print ("b_" + str(i) + " = " + str(b_n) + ", f(b_" + str(i) + ") = " + str(f_b_n))
    print ()


    # Conditions that determine & assign the next interval [a_n, b_n]
    if f_p_n < 0 : a_n = p_n
    else : a_n = a_n

    if f_p_n > 0 : b_n = p_n
    else : b_n = b_n

# TEST - For function to work, we must give our own choice of b
bisection(100)


''' I don't belive the b = 0.12 used to calculate N(t) and bisection(b) are equal.
 If you graph the function N(t)-1 it is visible that the intersection
 happens around 7.66. Therefore, you need to star with a b > 7.67'''

# 2. (15 pts) Create a Python function **newton(x)** that finds the root of 
# $N(t) - 1 = 0$ by Newton's method. The initial guess $p_0$ is $x$.

# <ul>
#     <li>Calculate the derivative $N'(t)$ manually and use it in the code.</li>
#     <li>Use an error bound $10^{-6}$. Note that the error size is estimated by $|p_{n+1} - p_n|$.</li>
#     <li>Allow at most 1000 iterations.</li>
#     <li>For each step, print $p_n$ and the estimation of the error $|p_n - p_{n-1}|$.</li>
# </ul>

def N_prime(t):
    n_0 = 3 * 10**(-5)
    n_i = 10**3
    b = 0.12
    k = math.log(n_i / n_0)

    if t == 0: return n_0 * b * k
    else: return n_0 * b * k * math.exp(-b*t) * math.exp(k*(1 - math.exp(-b*t)))


def newton(x) :
  # Setting up initial values
  p_n = x
  p_ni = p_n - ((N(p_n) - 1) / (N_prime(p_n)))
  error = abs(p_ni - p_n)
  i = 1

  print ("p_0 = " + str(p_n))

  while error > 10**(-6) :
    p_ni = p_n - ((N(p_n) - 1) / (N_prime(p_n)))
    error = abs(p_ni - p_n)

    # Printing p_n and the estimation of the error
    print ("p_" + str(i) + " = " + str(p_ni) + ", error = " + str(error))

    # Assigns next values
    p_n = p_ni
    i = i + 1

# TEST
newton(10)

# 3. (15 pts) Create a Python function **secant(x0, x1)** that finds the root of $N(t) - 1 = 0$ by secant method. $p_0 = x0$ and $p_1 = x1$.

# <ul>
#     <li>Use an error bound $10^{-6}$. You may estimate the error size by $|p_{n} - p_{n-1}|$.</li>
#     <li>Allow at most 1000 iterations.</li>
#     <li>For each step, print $p_n$ and the estimation of an error $|p_n - p_{n-1}|$.</li>
# </ul>

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
