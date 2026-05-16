# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1: Import the necessary packages using import statement.
Step2: Read the given csv file using read_csv() method and print the number of contents to be
displayed using df.head().
Step3: Import KMeans and use for loop to cluster the data.
Step4: Predict the cluster and plot data graphs.
Step5: Print the outputs and end the program
## Program:
```python
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by:Mokesh C
RegisterNumber:212225240088  
*/
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")
data.head()
data.info()
data.isnull()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss= [] 
for i in range(1,11):
kmeans=KMeans(n_clusters = i,init = "k-means++")
kmeans.fit(data.iloc[:,3:])
wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km=KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
1/4
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="black",label="clu
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="cyan",label="clust
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="yellow",label="cl
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="clust
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="green",label="clu
plt.legend()
plt.title("Customer Segments")
```

## Output:
<img width="764" height="1079" alt="WhatsApp Image 2026-05-16 at 4 16 17 PM" src="https://github.com/user-attachments/assets/adf99f22-95c4-4d07-8c58-4f21bb633aa6" />
<img width="551" height="322" alt="WhatsApp Image 2026-05-16 at 4 16 17 PM (1)" src="https://github.com/user-attachments/assets/7d9bb2e7-7a92-44c7-bb94-7a703d632234" />
<img width="715" height="258" alt="WhatsApp Image 2026-05-16 at 4 13 06 PM" src="https://github.com/user-attachments/assets/6f41440a-30bd-4764-b129-41615c95227d" />





## Result:
