"""
A common feature is that by using techniques from linear algebra, one may find a very nice approximation of given data (in a very big or even infinite-dimensional vector space) in a reasonable small vector space.

In this project, we investigate how to compress continuous sound data by using finitely many numbers.

Sound data is given as a continuous function $f(x)$. If we divide the time interval into reasonably small subintervals, then on each interval, the function $f(x)$ looks like a periodic function, because the sound is a vibration that propagates as an (in general very complicated) wave of pressure and a wave is periodic. From now on, for simplicity, assume that the period of $f(x)$ is $2\pi$ and we will assume that $f \in C[-\pi, \pi]$.

On $C[-\pi, \pi]$, the following formula yields an inner product:
$$\langle g, h\rangle = \frac{1}{\pi}\int_{-\pi}^{\pi}g(x)h(x)dx.$$
Furthermore, we know that the following set is an orthonormal set.
$$\mathcal{T}_n := \{\frac{1}{\sqrt{2}}, \sin x, \cos x, \sin 2x, \cos 2x, \cdots, \sin nx, \cos nx\}$$
Let $W_n$ be the sub vector space of $C[-\pi, \pi]$ spanned by $\mathcal{T}_n$. Then for any $f \in C[-\pi, \pi]$, its best approximation in $W_n$ is given by
$$S_n(x) := \langle f(x), 1\rangle \frac{1}{2} + \sum_{k=1}^n \langle f(x), \cos kx\rangle \cos kx + \sum_{k=1}^n \langle f(x), \sin kx\rangle \sin kx.$$

##### 1. (10 pts) Construct a function **Simpson(f, m)** which evaluates the definite integral $$\int_{-\pi}^{\pi}f(x)dx$$ with $m$ intervals by using the Simpson's rule. Note that $m$ has to be an even number.
"""

import numpy as np

# LaTeX rendering issue. The square above is T. Ex) Tn := {1/√2, sin} ...

#def f(x):# depends on this integrand function
#    return (x**2) # answer is 20.671
f = lambda x: -x**3*np.sin(x)

def Simpson(f, m):
  if m % 2 != 0:
    print("The amount of intervals must be even to run Simpson's Rule")
    return 0

  b = np.pi # bounds
  a = -np.pi
  h = (b - a) / m # size of each interval

  x = np.linspace(a, b, m + 1)
  y = f(x)

  S = y[0] + y[-1]                  # f(x0) + f(xn)
  S += 4 * np.sum(y[1:-1:2])        # 4 * odd
  S += 2 * np.sum(y[2:-2:2])        # 2 * even

  return (h / 3) * S

m = int(input("how many intervals would you like?"))
print(Simpson(f, m))
