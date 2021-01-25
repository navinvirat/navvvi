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
mdata1=data.iloc[i].to_numpy().reshape(28,28)
mdata2=data.iloc[j].to_numpy().reshape(28,28)
plt.imsave("mnist.png",mdata1,cmap="gray")
plt.imsave("mnist2.png",mdata2,cmap="gray")
con_image=np.concatenate((mdata1,mdata2),axis=-1)
plt.imsave("con2.png",con_image,cmap="gray")
img=cv.imread("ten2.png")
img1=cv.imread("zero8.png")
r1=cv.resize(img,(28,28))
print(r1.shape)
zero=[]
one=[]
for i in range(len(l1)):
    if(l1[i]==5):
        zero.append(i)
    if(l1[i]==1):
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
    plt.imsave("tens"+str(i)+".png",ten_rs,cmap="gray")
    plt.savefig("C:/Users/venky/OneDrive/Desktop/Smartail\Ex1/fifteen_mnist/tens"+str(i)+".png")
    




    

    




    





