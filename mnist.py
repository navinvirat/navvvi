import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2 as cv
d=pd.read_csv("mnist_train.csv")
d1=d
l1=d1["label"]
data1=d1.drop("label",axis=1)
l=d['label']
data=d.drop("label",axis=1)
plt.figure(figsize=(10,10))
i=0 
j=1

zero=[]
one=[]
for i in range(len(l1)):
    if(l1[i]==0):
        zero.append(i)
    if(l1[i]==3):
        one.append(i)
print(zero)
print(l[1])
for i in range(len(min(zero,one))):
    mdataz=data1.iloc[zero[i]].to_numpy().reshape(28,28)
    mdatao=data1.iloc[one[i]].to_numpy().reshape(28,28)
    ten=np.concatenate((mdatao,mdataz),axis=-1)
    plt.imsave("ten"+str(i)+".png",ten,cmap="gray")
    ten_r=cv.imread("ten"+str(i)+".png")
    ten_rs=cv.resize(ten_r,(28,28))
    #plt.imsave("tens"+str(i)+".png",ten_rs,cmap="gray")
    plt.imsave("C:/Users/venky/OneDrive/Desktop/Smartail\Ex1/30_mnist/tens"+str(i)+".png",ten_rs,cmap="gray")
    




    

    




    





