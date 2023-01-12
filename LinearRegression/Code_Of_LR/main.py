import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.subplot2grid((5, 2), (0, 0), rowspan=2)
data = pd.read_csv('data.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

x = np.hstack((np.ones((N, 1)), x, np.multiply(x, x)))
w = np.array([190, -97, 1.]).reshape(-1, 1)

# # Use Linear Regession
# x_T = np.transpose(x)
# invert_x = np.linalg.inv(np.dot(x_T, x))
# w = np.dot(np.dot(invert_x, x_T), y)

# Use Gradient descent
learning_rate = 0.01
r = np.dot(x, w) - y
v = np.zeros_like(w).reshape(-1, 1)
gamma = 0.9
while abs(np.average(r)) > 100:
    v[0] = gamma * v[0] + learning_rate * np.average(r * x[:, 0].reshape(-1, 1))
    w[0] = w[0] - v[0]
    v[1] = gamma * v[1] + learning_rate * np.average(r * x[:, 1].reshape(-1, 1))
    w[1] = w[1] - v[1]
    v[2] = gamma * v[2] + learning_rate * np.average(r * x[:, 2].reshape(-1, 1))
    w[2] = w[2] - v[2]

    r = np.dot(x, w) - y

predict = np.dot(x, w)
# plt.subplot2grid((5, 2), (3, 0), rowspan=2)
plt.plot(x[:, 1], predict, c="#E98EAD")
plt.xlabel('mét vuông')
plt.ylabel('giá')
plt.show()

x1 = 50
y1 = w[0] + w[1] * x1 + w[2] * x1 * x1
print('Giá nhà cho 50m^2 là : ', y1)
print(w)
print(abs(np.average(r)))




