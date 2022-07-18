import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

data=pd.read_csv('KNNAlgorithmDataset.csv')

x=data.iloc[:,[2,3]].values
y=data.iloc[:,1].values
for i in range(0,len(y)):
    if y[i]=='M':
        y[i]=1
    else:
        y[i]=0
y=y.astype('int')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#vẽ kết quả training
mtp.subplot(1,2,1)
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)
x1,x2=np.meshgrid(x1,x2)

# vẽ đường liền đã tô màu
mtp.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
mtp.xlim(x1.min(),x1.max())
mtp.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    mtp.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
mtp.title('KNN (Train set)')
mtp.legend()


#vẽ kết quả dự đoán
mtp.subplot(1,2,2)
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)
x1,x2=np.meshgrid(x1,x2)

# vẽ đường liền đã tô màu
mtp.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
mtp.xlim(x1.min(),x1.max())
mtp.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    mtp.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
mtp.title('KNN (Test set)')
mtp.legend()
mtp.savefig('demoKNN.png')