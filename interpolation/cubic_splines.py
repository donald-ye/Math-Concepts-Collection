"""3. Let $\mathrm{sample} = [x_0, x_1, \cdots, x_n]$ be a list of (not necessarily equally spaced) numbers such that $x_j < x_{j+1}$. Construct a python function **CubicSpline(sample)** 
  that calcuates the natural cubic spline function $S(x)$ with the sample points $\{(x_0, f(x_0)), (x_1, f(x_1)), \cdots, (x_n, f(x_n))\}$. Using this, find the natural cubic spline function $S(x)$ with:
    
    (a) evenly spaced sample points $x_0 = -5, x_1, x_2, \cdots, x_9 = 5$,

    (b) $x_0 = 5\cos(\frac{19}{20}\pi), x_1 = 5\cos(\frac{17}{20}\pi), x_2 = 5\cos(\frac{15}{20}\pi), \cdots, x_9 = 5\cos(\frac{1}{20}\pi)$, and sketch the graph of $f(x)$ and $S(x)$ with 
    these two sets of sample points on the same plane (for $-5 \le x \le 5$).
"""

def CubicSpline(x_vals, y_vals):
  n = len(x_vals) - 1
  h = np.diff(x_vals)

  a = np.zeros(n) # a-values
  for i in range(1, n):
    a[i] = (3/h[i]) * (y_vals[i+1] - y_vals[i]) - (3/h[i-1]) * (y_vals[i] - y_vals[i-1])

  A = np.zeros((n+1, n+1)) # set up matrix
  b = np.zeros(n+1)
  A[0,0], A[n,n] = 1, 1  # boundary conditions

  for i in range(1, n):
      A[i, i-1] = h[i-1]
      A[i, i] = 2 * (h[i-1] + h[i])
      A[i, i+1] = h[i]
      b[i] = a[i]

  c = np.linalg.solve(A, b) # solving for c

  b = np.zeros(n)
  d = np.zeros(n)
  a = y_vals[:-1]

  splines = []
  for j in range(n):
      b[j] = (y_vals[j+1] - y_vals[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
      d[j] = (c[j+1] - c[j]) / (3 * h[j])
      coeffs = [a[j], b[j], c[j], d[j]]
      splines.append(P(coeffs))

  return splines


def fx(x): # computes function
  return 1 / (1 + x**2)

#(a)
x_vals_approx = np.linspace(-5,5,10)
y_vals_approx = fx(x_vals_approx)
spline_approx = CubicSpline(x_vals_approx, y_vals_approx)

print("Cubic Splines for Approximated Values:")
for i, poly in enumerate(spline_approx):
    print(f"Spline {i}: {poly}")

#(b)
x_vals_precise = np.array([5 * np.cos((k / 20) * np.pi) for k in range(19, 0, -2)])
y_vals_precise = 1 / (1 + x_vals_precise**2)
spline_precise = CubicSpline(x_vals_precise, y_vals_precise)

print("\n")
print("Cubic Splines for Precise Values: ")
for i, poly in enumerate(spline_precise):
    print(f"Spline {i}: {poly}")

# graph sketch

def plot_graph():
  plt.figure(figsize =(10,7))
  x_plot = np.linspace(-5, 5, 1000)
  y_actual = fx(x_plot)

  #evenly spaced points
  plt.scatter(x_vals_approx, y_vals_approx, color="blue", marker="x", label="Approximately Spaced Points")

  #precisly spaced points
  plt.scatter(x_vals_precise, y_vals_precise, color="orange", marker="x", label="Precisly Spaced Points")

  for i in range(len(spline_approx)):
      x_range = np.linspace(x_vals_approx[i], x_vals_approx[i + 1])
      y_range = spline_approx[i](x_range - x_vals_approx[i])  # Evaluate spline at transformed x
      plt.plot(x_range, y_range, color="red", label="Approximate Cubic Spline" if i == 0 else "")

    # Plot cubic splines for precisely spaced points
  for i in range(len(spline_precise)):
      x_range = np.linspace(x_vals_precise[i], x_vals_precise[i + 1])
      y_range = spline_precise[i](x_range - x_vals_precise[i])
      plt.plot(x_range, y_range, color="green", label="Precise Cubic Spline" if i == 0 else "")

  #settings
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("Cubic Splines")
  plt.legend(loc="upper left", fontsize=8)
  plt.show()

plot_graph()

"""4. In computer graphics, <b>vectorization</b> is the conversion of raster graphics into vector graphics. Roughly, raster graphics is a graphic format that consists of a collection of colored dots. 
      Examples include JPG, GIF, and PNG formats. Raster graphics are resolution dependent, meaning they cannot scale up to an arbitrary resolution without loss of apparent quality. (If you zoom in a 
      jpg file, then you can see many colored squares.) On the other hand, vector graphics are defined in terms of points connected by lines and curves to form polygons and other shapes. Typical examples 
      are font files, SVG, and PDF. Because they encode relevant mathematical data, it could be easily scaled up without any loss of quality.Vectorization or image tracing is a method to convert a given 
      raster graphic data into a vector graphic format. This project will investigate one of the simplest kinds of image tracing by using the spline function. Consider the following image of a turtle:

      ![turtle.jpg](https://drive.google.com/uc?id=1oOa6agPnBA4hcGsa9nDJ8QXp4I9yeL-U)

  What you need to do is trace the contour of the upper part of the turtle's body by using natural cubic spline functions. More precisely, you should calculate a natural cubic spline function $S(x)$ that describes the upper outline of the turtle, starting from the left end of its shell to the rightmost point to its eye.
  
  <ol>
      <li>By using some graphic tool such as Microsoft Paint or Photoshop, find the coordinates of several sample points.</li>
      <li>Compute the natural cubic spline function connecting the sample points.</li>
      <li>Plot your spline function over the image. </li>
  </ol>
  
  Your graph of the spline function should have no significant error (I know this is somewhat subjective, but...) compared to the outline of the original image. 10 pts for the correctness of the code. The remaining 10 points are for the best performance, in terms of the <em>number of sample points</em>. For example, the team using the smallest number of sample points will get 10 pts, the next team will get 9 pts, etc. Keep in mind that your spline should not have any significant difference from the original image. It is important which sample points you use. Try several examples and get some lessons.
"""

x = np.array([241.8, 329, 609, 889, 1175.4, 1248.6, 1401.7, 1506.6])
y = np.array([-654.3, -607.6, -408.8 ,-307.4, -313.5, -344.3, -263.3, -295.5])

turtle_spline = CubicSpline(x, y)

print("Cubic Splines for Turtle:")
for i, poly in enumerate(turtle_spline):
    print(f"Spline {i}: {poly}")

def plot_graph():
  plt.figure(figsize =(15,7))
  x_plot = np.linspace(-5, 5, 1000)
  y_actual = fx(x_plot)

  plt.axis("equal")
  plt.gca().set_aspect('equal', adjustable='datalim')


  #evenly spaced points
  plt.scatter(x, y, color="blue", marker="x", label="Approximately Spaced Points")


    # Plot cubic splines for precisely spaced points
  for i in range(len(turtle_spline)):
      x_range = np.linspace(x[i], x[i + 1])
      y_range = turtle_spline[i](x_range - x[i])
      plt.plot(x_range, y_range, color="green", label="Turtle Line" if i == 0 else "")

  #settings
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("Turtle Splines")
  plt.legend(loc="upper left", fontsize=10)
  plt.show()

plot_graph()
