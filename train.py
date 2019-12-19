import pandas as pd
import numpy as np
import logging 
import math

from sklearn.preprocessing import StandardScaler,MinMaxScaler # to scale data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score

# NN model
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import SGD #test keras

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 200)

train = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

logger.info('Reading input files')
files = ["data/df_18_6638.csv","data/df_18_6648.csv"]
frames = []
for i,f in enumerate(files):
    frames.append(pd.read_csv(f))
    #print(frames[i].head())
    #print(frames[i].columns)
    #print(frames[i].shape)

logger.info('Concatenating DataFrames')        
df = pd.concat(frames)

#I want only beam mode >5 <12
logger.info("Cleaning beam mode: keep >5 and <12")
df = df[(df['BMode']>5) & (df['BMode']<12)]

logger.info('Converting DataFrame into np arrays')
X = df[['ATLASlumi', 'CMSlumi', 'IntB1', 'Ene', 'Betastar_IP1', 'Betastar_IP5', 'Xing_IP1','Xing_IP5', 'BMode']].values
y = df['LifetimeB1'].values

# scaling
logger.info('Scaling features')
#sc = MinMaxScaler()  
sc = StandardScaler()
X = sc.fit_transform(X)
y = y.reshape(-1,1)
y = sc.fit_transform(y)

logger.info('Train and test splitting')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

logger.info('Model definition and compilation')
model = Sequential() # creating model sequentially (each layer takes as input output of previous layer)
model.add(Dense(10, input_dim=9, activation='relu', kernel_initializer='normal')) # Dense: fully connected layer
model.add(Dense(80, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal')) # chiara: check what's the best activation function for single-value output
# loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')
# training 
logger.info('Model training')
history = model.fit(X_train, y_train, epochs=100, batch_size=50, # it was 100 epochs
                    validation_data = (X_test,y_test)) # show accuracy on test data after every epoch
# Prediction
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

model.save('models/my_model.h5')

'''
logger.info('Plotting accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''

logger.info('Plotting loss function')
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

logger.info('Plotting predictions')
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
