# example on brillian

import numpy as np
import matplotlib.pyplot as plt

y = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
x = np.array([ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
colors=[80,81,82,83,84,85,86,87,88,89,90,91,92]
plt.scatter(x, y, c=colors, cmap='winter', s=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50], alpha=0.5)

x1 = np.linspace(40,80,2)
y1 = 1.77839211 * x1 + 60.91228332

plt.plot(x1, y1)

plt.show()
