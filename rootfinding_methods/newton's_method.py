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
