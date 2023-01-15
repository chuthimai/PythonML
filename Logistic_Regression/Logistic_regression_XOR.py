import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv("data_XOR.csv").values
x = data[:, :-1].reshape(-1, 2)
y = data[:, -1].reshape(-1, 1)

x_true = x[y[:, 0] == 1]
x_false = x[y[:, 0] == 0]

plt.scatter(x_true[:, 0], x_true[:, 1], c="#03C988", label="True")
plt.scatter(x_false[:, 0], x_false[:, 1], c="#58287F", label="False")
plt.legend()
plt.xlabel("A")
plt.ylabel("B")

# Thêm cột 1 vào dữ liệu x
x = np.hstack((np.ones((4, 1)), x))

w = np.array([0., 0.1, 0.1]).reshape(-1, 1)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.1

for i in range(1, numOfIteration):
    # Tính giá trị dự đoán
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
    # Gradient descent
    w = w - learning_rate * np.dot(x.T, y_predict - y)
    print(cost[i])

# Vẽ đường phân cách.
t = 0.5
plt.plot((0, 1), (-(w[0] + 0 * w[1] + np.log(1 / t - 1)) / w[2], -(w[0] + 1 * w[1] + np.log(1 / t - 1)) / w[2]), 'g')
plt.show()
print(w)






