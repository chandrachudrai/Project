#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
np.random.seed(1)


# In[2]:


dt=pd.read_csv("dataset.csv")


# In[3]:


X=dt.drop(["diagnosis"],axis=1)
Y=dt["diagnosis"]


# In[4]:


M=X.mean()
M=np.array(M).reshape((1,5))
SD=np.std(X)


# In[5]:


SD=np.array(SD).reshape(1,5)


# In[6]:


X=np.array(X)
X=np.divide(X-M,SD)
X=X.T
Y=np.array(Y).reshape((1,569))


# In[7]:


n_x=np.shape(X)[0]
n_h=4
n_y=np.shape(Y)[0]
m=np.shape(Y)[1]


# In[8]:


#model parameters
W1=np.random.randn(n_h,n_x)*0.01
b1=np.zeros((n_h,1))
W2=np.random.randn(n_y,n_h)*0.01
b2=np.zeros((n_y,1))
print(W1)
print(b1)
print(W2)
print(b2)


# In[9]:


learning_rate=1
costs=[]
t=25000
for i in range(0,t):
    #activation
    Z1=np.dot(W1,X) + b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1) + b2
    A2=1/(1+np.exp(-Z2))
    A2.reshape((1,X.shape[1]))
    #loss
    loss=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
    cost=-np.sum(loss)/m
    cost=float(np.squeeze(cost))
    #gradient
    dZ2=A2-Y
    dW2=np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
    #updation
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    costs.append(cost)
    if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
print("W1= "+str(W1))
print("b1= "+str(b1))
print("W2= "+str(W2))
print("b2= "+str(b2))

print("dW1= "+str(dW1))
print("db1= "+str(db1))
print("dW2= "+str(dW2))
print("db2= "+str(db2))


# In[10]:


x = np.linspace(0, t, t)
y=costs
plt.plot(x,y,'r')


# In[11]:


Z1=np.dot(W1,X)+b1
A1=np.tanh(Z1)
Z2=np.dot(W2,A1)+b2
A2=1/(1+np.exp(-Z2))
predictions=(A2>0.5)


# In[12]:


print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# In[ ]:





# In[ ]:




