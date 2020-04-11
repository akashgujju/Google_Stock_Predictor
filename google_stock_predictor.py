import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

dataset = pd.read_csv("Google_Stock_Price_Train.csv")
X = dataset.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_sc = sc.fit_transform(X)


X_train=[]
y_train=[]
for i in range(60,len(X)):
    X_train.append(X_sc[i-60:i,0])
    y_train.append(X_sc[i,0])
    
    
X_train,y_train = np.array(X_train),np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,batch_size=32,epochs=80)

dataset2 = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset2.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset['Open'], dataset2['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset2) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
