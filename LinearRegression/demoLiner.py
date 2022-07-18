
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

x = np.array([190,195,198,200,204,205])
y = np.array([5,4,2,8,6,1])

plt.subplot(2,2,1)
plt.plot(x,y,marker='*',ms=10)
plt.grid()
plt.title("Đồ thị gấp khúc")

plt.subplot(2,2,2)
plt.bar(x,y)
plt.title("Đồ thị cột dọc")

plt.subplot(2,2,3)
plt.barh(x,y)
plt.title("Đồ thị cột ngang")

plt.savefig("Demo0")

plt.subplot(1,1,1)
colors = np.array([50,60,70,80,90,100])
plt.scatter(x,y,s=[1000,2000,3000,4000,5000,6000],alpha=0.8,c=colors,cmap='summer_r')
plt.colorbar()
plt.title("Đồ thị phân tán")
plt.savefig("Demo1")

plt.subplot(1,1,1)
plt.plot(x,y,'m-.')
plt.title("Biểu đồ đường")
plt.savefig("Demo2")

plt.subplot(1,1,1)
plt.pie(y,startangle=90,explode=[0,0.5,0,0,0,0])
plt.title("Biểu đồ tròn")
plt.savefig("Demo3")


