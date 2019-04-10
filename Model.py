import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf 

np.random.seed(1337)   # to reproduce results 

# load images into a dataframe 
images_dataframe = pd.DataFrame
images_dataframe = pd.read_csv("train.csv", sep=',')

#prepare labels 
train_y = images_dataframe['label']
k = pd.get_dummies(train_y,prefix='number')

images_dataframe = images_dataframe / 255
images_dataframe = images_dataframe.drop(['label'],axis = 1)
train_x = images_dataframe


# Build the sequential model 
model = Sequential()
model.add(Dense(784, activation='relu', input_shape=(784,), kernel_initializer = 'normal'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation= 'softmax',  kernel_initializer = 'normal'))
model.compile(optimizer = 'adam', loss ='mean_squared_error'  , metrics = ['acc'])
model.fit(train_x, k, validation_split=0.8 , epochs=5)

#load the test images
test = pd.DataFrame
test = pd.read_csv("test.csv", sep=',')
predictions = model.predict(test)

#store predictions to a numpy array 
k = np.zeros((28000,))
for i in range(27999):
    k[i] = np.argmax(predictions[i])


# store the predictions from numpy array into dataframe      
f = pd.DataFrame  
f = pd.read_csv('sample_submission.csv',sep = ',')
f['Label']  = k 

# Put the predicted labels into csv file 
f.to_csv('sample.csv')

