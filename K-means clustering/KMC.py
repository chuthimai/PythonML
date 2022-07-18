import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

data=pd.read_csv('Mall_Customers.csv')
# print(data.head(5))

x=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

# wcss_list=[]
# for i in range(1,11):
#     kmeans=KMeans(n_clusters=i,random_state=42)
#     kmeans.fit(x)
#     wcss_list.append(kmeans.inertia_)
# mtp.plot(range(1,11),wcss_list,c='red')
# mtp.show()

kmeans=KMeans(n_clusters=5,random_state=42)
y_pred=kmeans.fit_predict(x)

mtp.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,cmap='Blues',label='Cluster1')
mtp.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,cmap='Accent',label='Cluster2')
mtp.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,cmap='BrBG',label='Cluster3')
mtp.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100,cmap='Greens',label='Cluster4')
mtp.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100,cmap='Purples',label='Cluster5')

mtp.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,cmap='red',label='Centroid')
mtp.title('Clusters of customers')
mtp.xlabel('Annual Income($)')
mtp.ylabel('Spending Score(1-100)')
mtp.legend()
mtp.savefig('K-means demo.png')



