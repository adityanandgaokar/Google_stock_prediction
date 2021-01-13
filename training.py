
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


train_data = pd.read_csv('D:/Projects/Google_Stock_Prediction/Google_Stock_Price_Train.csv')

X = train_data.iloc[:, 1].values
X = X.reshape(-1, 1)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(X)

x_train = []
y_train = []


for i in range(60, 1258):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
    
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)


model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))


model.compile(optimizer = 'adam', loss ='mean_squared_error')

model.fit(x_train, y_train, epochs = 100, batch_size = 32)

model.save('D:/Projects/Google_Stock_Prediction/LSTM.h5')
