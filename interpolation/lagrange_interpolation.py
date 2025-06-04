"""
For the first problem, $f(x) = \frac{1}{1+x^2}$. We will compute several approximations of $f(x)$ on $[-5, 5]$, coming from various interpolations of sample points on the graph of $f(x)$.

1. (10 pts) Let $\mathrm{sample} = [x_0, x_1, \cdots, x_n]$ be a list of (not necessarily equally spaced) numbers such that $x_j < x_{j+1}$. Construct a python function **Lagrange(sample)** 
that calculates the polynomial interpolation $p_n(x)$ of $\{(x_0, f(x_0)), (x_1, f(x_1)), \cdots, (x_n, f(x_n))\}$ on $[x_0, x_n]$ using <em>Lagrange interpolation</em>.
Using this, find the polynomial interpolation $p_{9}(x)$ of $\{(x_0, f(x_0)), (x_1, f(x_1)), \cdots, (x_9, f(x_9))\}$ of degree $\le 9$ with:

(a) evenly spaced sample points $x_0 = -5, x_1, x_2, \cdots, x_9 = 5$,

(b) $x_0 = 5\cos(\frac{19}{20}\pi), x_1 = 5\cos(\frac{17}{20}\pi), x_2 = 5\cos(\frac{15}{20}\pi), \cdots, x_9 = 5\cos(\frac{1}{20}\pi)$,
and sketch the graph of $f(x)$ and $p_{9}(x)$ with these two sets of sample points on the same plane (for $-5 \le x \le 5$).
"""

import numpy as np
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt

def Lagrange(x_vals, y_vals):

  n = len(x_vals)
  poly = P([0])

  for i in range(n):
    term = P([y_vals[i]])
    base = P([1])

    for j in range(n):
      if i != j:
        base *= P([-x_vals[j], 1]) / (x_vals[i] - x_vals[j])

    poly += term * base
  return poly


def fx(x): # computes function
  return 1 / (1 + x**2)

#(a)
x_vals_approx = np.linspace(-5,5,10)
y_vals_approx = fx(x_vals_approx)

interp_approx = Lagrange(x_vals_approx, y_vals_approx)
print("Lagrange Polynomial at Approximately Spaced Points: \n", interp_approx, "\n")

#(b)
x_vals_precise = np.array([5 * np.cos((k / 20) * np.pi) for k in range(19, 0, -2)])
y_vals_precise = 1 / (1 + x_vals_precise**2)

interp_precise = Lagrange(x_vals_precise, y_vals_precise)
print("Lagrange Polynomial at Precisly Spaced Points: \n", interp_precise)

#graph sketch
def plot_graph():
  plt.figure(figsize =(10,7))

  x_plot = np.linspace(-5, 5, 1000)
  y_actual = fx(x_plot)
  y_interp_approx = interp_approx(x_plot)
  y_interp_precise = interp_precise(x_plot)

  #evenly spaced points
  plt.scatter(x_vals_approx, y_vals_approx, color="blue", marker="x", label="Approximately Spaced Points")
  plt.plot(x_plot, y_interp_approx, label='Lagrange Interpolation Approximately Spaced', color="red")

  #precisly spaced points
  plt.scatter(x_vals_precise, y_vals_precise, color="orange", marker="x", label="Precisly Spaced Points")
  plt.plot(x_plot, y_interp_precise, label='Lagrange Interpolation Precisly Spaced', color="green")

  #settings
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("Lagrange Interpolation")
  plt.legend(loc="upper left", fontsize=5.5)
  plt.show()

plot_graph()

def fx(x):  # Function f(x) = 1 / (1 + x^2)
    return 1 / (1 + x**2)

def Lagrange(x_vals):
    y_vals = fx(x_vals)  # Compute function values inside the function
    n = len(x_vals)
    poly = P([0])  # Initialize polynomial to 0

    for i in range(n):
        term = P([y_vals[i]])  # Start with f(x_i)
        base = P([1])  # Start with 1 for multiplication

        for j in range(n):
            if i != j:
                base *= P([-x_vals[j], 1]) / (x_vals[i] - x_vals[j])  # Construct Lagrange basis

        poly += term * base  # Add term to polynomial
    return poly

# (a) Evenly spaced sample points
x_vals_approx = np.linspace(-5, 5, 10)
interp_approx = Lagrange(x_vals_approx)

# (b) Precisely spaced sample points
x_vals_precise = np.array([5 * np.cos((k / 20) * np.pi) for k in range(19, 0, -2)])
interp_precise = Lagrange(x_vals_precise)

# Graph sketch including f(x)
def plot_graph():
    plt.figure(figsize=(10, 7))
    x_plot = np.linspace(-5, 5, 1000)
    y_actual = fx(x_plot)  # Actual function values
    y_interp_approx = interp_approx(x_plot)
    y_interp_precise = interp_precise(x_plot)

    # Plot f(x)
    plt.plot(x_plot, y_actual, label='Actual f(x)', color="black", linestyle="dashed")

    # Evenly spaced points
    plt.scatter(x_vals_approx, fx(x_vals_approx), color="blue", marker="x", label="Approximately Spaced Points")
    plt.plot(x_plot, y_interp_approx, label='Lagrange Interpolation (Approx)', color="red")

    # Precisely spaced points
    plt.scatter(x_vals_precise, fx(x_vals_precise), color="orange", marker="x", label="Precisely Spaced Points")
    plt.plot(x_plot, y_interp_precise, label='Lagrange Interpolation (Precise)', color="green")

    # Settings
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Lagrange Interpolation vs. f(x)")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()

plot_graph()
