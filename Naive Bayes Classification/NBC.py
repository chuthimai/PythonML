import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

data=pd.read_csv('Social_Network_Ads.csv')
# print(data.head(5))

x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#vẽ training set
mtp.subplot(1,2,1)
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
arr1=np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01)
arr2=np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01)
x1,x2=np.meshgrid(arr1,arr2)
mtp.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.5,cmap=ListedColormap(('blue','green')))
mtp.xlim(x1.min(),x1.max())
mtp.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    mtp.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap(('blue','green'))(i),label=j)

mtp.title('NBC (training set)')
mtp.legend()

#vẽ test set
mtp.subplot(1,2,2)
x_set,y_set=x_test,y_test
arr1=np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01)
arr2=np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01)
x1,x2=np.meshgrid(arr1,arr2)
mtp.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('pink','purple')))
mtp.xlim(x1.min(),x1.max())
mtp.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    mtp.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap(('pink','purple'))(i),label=j)

mtp.title('NBC (test set)')
mtp.legend()
mtp.savefig('NBC.png')
