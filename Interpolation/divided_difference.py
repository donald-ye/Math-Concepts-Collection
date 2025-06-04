"""
  2. Let $\mathrm{sample} = [x_0, x_1, \cdots, x_n]$ be a list of (not necessarily equally spaced) numbers such that $x_j < x_{j+1}$. Construct a python function 
  **DivDiff(sample)** that calcuates the polynomial interpolation $p_n(x)$ of $\{(x_0, f(x_0)), (x_1, f(x_1)), \cdots, (x_n, f(x_n))\}$ using <em>divided differences</em>.
  Using this, find the polynomial interpolation $p_{9}(x)$ of $\{(x_0, f(x_0)), (x_1, f(x_1)), \cdots, (x_9, f(x_9))\}$ of degree $\le 9$ with:
    (a) evenly spaced sample points $x_0 = -5, x_1, x_2, \cdots, x_9 = 5$,

    (b) $x_0 = 5\cos(\frac{19}{20}\pi), x_1 = 5\cos(\frac{17}{20}\pi), x_2 = 5\cos(\frac{15}{20}\pi), \cdots, x_9 = 5\cos(\frac{1}{20}\pi)$, and sketch the graph of $f(x)$ 
      and $p_{9}(x)$ with these two sets of sample points on the same plane (for $-5 \le x \le 5$).
"""

def DivDiff(x_vals, y_vals):
  n = len(x_vals)
  coef = y_vals.copy()

  for i in range(1, n):
    coef[i:] = (coef[i:] - coef[i-1]) / (x_vals[i:] - x_vals[i-1])

  poly = P([coef[0]])
  term = P([1])

  for i in range(1, n):
    term *= P([-x_vals[i-1], 1])
    poly += coef[i] * term

  return poly


def fx(x): # computes function
  return 1 / (1 + x**2)

#(a)
x_vals_approx = np.linspace(-5,5,10)
y_vals_approx = fx(x_vals_approx)

interp_approx = DivDiff(x_vals_approx, y_vals_approx)
print("Divided Difference Polynomial at Approximately Spaced Points: \n", interp_approx, "\n")

#(b)
x_vals_precise = np.array([5 * np.cos((k / 20) * np.pi) for k in range(19, 0, -2)])
y_vals_precise = 1 / (1 + x_vals_precise**2)

interp_precise = DivDiff(x_vals_precise, y_vals_precise)
print("Divided Difference Polynomial at Precisly Spaced Points: \n", interp_precise)

#graph sketch
def plot_graph():
  plt.figure(figsize =(10,7))
  x_plot = np.linspace(-5, 5, 1000)
  y_actual = fx(x_plot)
  y_interp_approx = interp_approx(x_plot)
  y_interp_precise = interp_precise(x_plot)

  #evenly spaced points
  plt.scatter(x_vals_approx, y_vals_approx, color="blue", marker="x", label="Approximately Spaced Points")
  plt.plot(x_plot, y_interp_approx, label='Divided Difference Approximately Spaced', color="red")

  #precisly spaced points
  plt.scatter(x_vals_precise, y_vals_precise, color="orange", marker="x", label="Precisly Spaced Points")
  plt.plot(x_plot, y_interp_precise, label='Divided Difference Precisly Spaced', color="green")

  #settings
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("Divided Difference")
  plt.legend(loc="upper left", fontsize=6)
  plt.show()

plot_graph()
