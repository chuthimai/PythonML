import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

data=pd.read_csv('LinearRegression-Sheet1.csv')
#print(data.head(5))
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
x_pred=reg.predict(x_train)

mtp.scatter(x_train,y_train,c='green')
mtp.plot(x_test,y_pred,c='red')
mtp.scatter(x_test,y_test,c='pink')
mtp.savefig('demoLinearRegression.png')