#!/usr/bin/env python
# coding: utf-8

# # Classification Using ANN(Artifical Nural Network)

# ### --------Install Packages------- 

# 
# 
# Tensorflow --> For Fast Numeric Computation
# 
# conda create -n tensorflow
#   
# 
# Keras -->  Wrap up of tensorflow/THeano which can reduce the size of code
# 
# pip install --upgrade keras 

# In[1]:


#For Suppressing TensorFlow warnings of an older version
import warnings
warnings.simplefilter("ignore")


# ### --------Pre Processing Data -------

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


# In[3]:


dataset=pd.read_csv('musk_csv.csv')
dataset.head()


# #### Separating the Dependent , Independent , None Relative variables
# 
# None Relavent Variables :-
# 
# 
# ID,Molecule_Name,Conformation_Name

# In[4]:


#Features Variables(Independent Variables)
X=dataset.iloc[:,3:169].values
print(X)


# In[5]:


#Target Variable(Dependent Variable)
Y=dataset.iloc[:,169].values
print(Y)


# 
# #### All data are in Numerical form ( No categorical Data)
# 
# We can directly process to splitting the data

# #### Splitting into random 80:20 train test data

# In[6]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
print("X_Train")
print(X_train.shape)
print("Y_Train")
print(Y_train.shape)
print("X_Test")
print(X_test.shape)
print("Y_Test")
print(Y_test.shape)


# #### Features Scalling 
# 
# Avoid one independent variable dominating another one
# 
# Avoid Biasing of Independent Variables
# 
# Make Computation Easy
# 
# 

# In[7]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print("X_train")
print(X_train)
print("X_test")
print(X_test)


# ### -------- Create A Model ----------

# In[8]:


#Initialization the ANN
classifier=Sequential()
    
#Adding ip layer And first hidden layer
classifier.add(
    	Dense(activation="relu", input_dim=166, units=83,
               kernel_initializer="uniform"
    		)
    	)
#units=no of nodes in hidden layer --> Perameter Tuning or avg(no of nodes in ip layer,op layer)
#activation --> hidden layer (rectify fn) and op layer(sigmoid fn)
#input_dim=no of nodes in ip layer --> no of independent variables
    
#Adding 2nd Hidden Layer
classifier.add(
    	Dense(activation="relu", units=83, 
               kernel_initializer="uniform"
    		)
    	)
        
    
#Adding Op Layer
classifier.add(
    	Dense(activation="sigmoid", units=1, 
               kernel_initializer="uniform"
    		)
    	)
#output_dim=1--> Binary Classification
#activation --> sigmoid --> Probability of binary outcome
    
   
        
#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
# loss--> logarithmic loss
# binary -->binary_crossentropy
# accuracy matrix is use to compute the optimal value of weight in next itteration
    

    
    
    
classifier.summary()    
  


# #### Train a Model 

# In[9]:



history=classifier.fit(X_train,Y_train,batch_size=32,epochs=10,validation_split=.33)
#epoch --> number of time whole dataset is passed from model
#batch size--> number of observation after which you want to update the wight
#history will be used in graph plot


# In[10]:


#Saving A model

classifier.save('my_model.h5')


# ### -----------------Post Processing Of Data---------------

# In[11]:


y_pred=classifier.predict(X_test)
#it returns probability but we need binary value 0/1
#0-->Non Musk
#1-->Musk
print(y_pred)


# #### Thresholding 
# 
# value>0.5 --> 1
# 

# In[12]:


y_pred=(y_pred>0.50 )
#It Will return boolean
y_pred=y_pred.astype('int64')
#Covert in int
print(y_pred)


# ### ---------Result Testing And Analysis---------------

# In[13]:


cm=confusion_matrix(Y_test,y_pred)
print("Confusion Matrix")
print(cm)
print("Analysis OF Confusion Martix")
print("True Positive : "+str(cm[1][1]))
print("False Positive : "+str(cm[0][1]))
print("True Negative : "+str(cm[0][0]))
print("False Negative : "+str(cm[1][0]))


# In[14]:


print(classification_report(Y_test,y_pred))


# ### -----------Graph OF accuracy And loss--------

# In[15]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[16]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### -----Conclusion -----------
# 
# We are getting almost similar accuracy for both training and as well as test data set
# 
# 
# Our Model Is Capable of solving given buissness Problem

# In[ ]:





# In[ ]:




