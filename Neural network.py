#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
np.random.seed(3)


# In[48]:


dt=pd.read_csv("dataset.csv")


# In[49]:


X=dt.drop(["diagnosis"],axis=1)
Y=dt["diagnosis"]


# In[50]:


M=X.mean()
M=np.array(M).reshape((1,5))
SD=np.std(X)
Y.shape


# In[51]:


SD=np.array(SD).reshape(1,5)


# In[52]:


X=np.array(X)
X=np.divide(X-M,SD)
X=X.T
Y=np.array(Y).reshape((1,569))


# In[53]:


layers=[5,12,7,5,1]
L=len(layers)
S=L-1
m=Y.shape[1]


# In[54]:


parameters={}
for l in range (1,L):
    parameters["W"+str(l)]=2*np.random.randn(layers[l],layers[l-1])-1
    parameters["b"+str(l)]=np.zeros((layers[l],1))


# In[55]:


Zi={}
Ai={}
At=X
Ai["A"+str(0)]=X


# In[56]:


dW={}
db={}


# In[57]:


learning_rate=0.5
costs=[]
t=10000
for i in range(0,t):
    
    #forward prop
    for j in range(1,S):
        Z=np.dot(parameters["W"+str(j)],At)+parameters["b"+str(j)]
        Ai["A"+str(j)]=np.tanh(Z)
        At=Ai["A"+str(j)]
    Z=np.dot(parameters["W"+str(S)],At)+parameters["b"+str(S)]
    AL=1/(1+np.exp(-Z))
    
    #cost calculation
    loss=np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),(1-Y))
    cost=-np.sum(loss)/m
    cost=float(np.squeeze(cost))
    
    #back prop
    dAL=-(np.divide(Y, AL)-np.divide(1-Y,1-AL))
    dZ=np.multiply(dAL,AL*(1-AL))
    dW["dW"+str(S)]=np.dot(dZ,Ai["A"+str(S-1)].T)/m
    db["db"+str(S)]=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(parameters["W"+str(S)].T,dZ)
    
    for l in reversed(range(1,S)):
        dZ=np.multiply(dA_prev,1-np.power(Ai["A"+str(l)],2))
        dW["dW"+str(l)]=np.dot(dZ,Ai["A"+str(l-1)].T)/m
        db["db"+str(l)]=np.sum(dZ,axis=1,keepdims=True)/m
        dA_prev = np.dot(parameters["W"+str(l)].T,dZ)
    
    for k in range(1,L):
        parameters["W" + str(k)] = parameters["W" + str(k)]- learning_rate*dW["dW"+str(k)]
        parameters["b" + str(k)] = parameters["b" + str(k)]- learning_rate*db["db"+str(k)]
    costs.append(cost)
    if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
print ("dW1 = "+ str(dW["dW1"]))
print ("db1 = "+ str(db["db1"]))
print ("dW2 = "+ str(dW["dW2"]))
print ("db2 = "+ str(db["db2"]))


# In[58]:


x = np.linspace(0, t, t)
y=costs
plt.plot(x,y,'r')


# In[59]:


At=X
for j in range(1,S):
    Zi["Z"+str(j)]=np.dot(parameters["W"+str(j)],At)+parameters["b"+str(j)]
    Ai["A"+str(j)]=np.tanh(Zi["Z"+str(j)])
    At=Ai["A"+str(j)]
Zi["Z"+str(S)]=np.dot(parameters["W"+str(S)],At)+parameters["b"+str(S)]
AL=1/(1+np.exp(-Zi["Z"+str(S)]))
predictions=(AL>0.5)


# In[60]:


print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# In[ ]:




