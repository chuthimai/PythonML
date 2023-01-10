import matplotlib.pyplot as plt
import numpy as np

plot_function = plt.subplot2grid((5, 2), (0, 0), rowspan=2)
x = np.linspace(-10, 10, 100)
y = x**2 + 2*x + 5
plt.xlabel("x")
plt.ylabel("y")
plt.title("y=f(x)=x^2+2x+5")
plt.plot(x, y)

plot_GD = plt.subplot2grid((5, 2), (3, 0), rowspan=2)
learning_rate = 0.01
x0 = -2
x = x0 - learning_rate*(2*x0+2)
y = []
while abs(x - x0) >= 0.0001:
    y0 = x ** 2 + 2 * x + 5
    y.append(y0)
    x0 = x
    x = x0 - learning_rate * (2 * x0 + 2)
plt.xlabel("Lần lặp")
plt.ylabel("f(x)")
plt.title("Gradient descent")
plt.plot(range(0, len(y), 1), y)
plt.show()




