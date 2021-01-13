import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('D:/Projects/Google_Stock_Prediction/Google_Stock_Price_Train.csv')

test_data = pd.read_csv('D:/Projects/Google_Stock_Prediction/Google_Stock_Price_Test.csv')

stock_price = test_data.iloc[:, 1:2].values

total_stock = pd.concat((train_data['Open'], test_data['Open']), axis = 0)

##print(len(total_stock))

print(len(train_data))

data = total_stock[len(train_data)-60:].values

data = data.reshape(-1,1)

scaler = MinMaxScaler(feature_range = (0,1))
X = train_data.iloc[:, 1].values
X = X.reshape(-1, 1)
scaled_data = scaler.fit_transform(X)

data = scaler.transform(data)

x_test = []

for i in range(60,80):
    x_test.append(data[i-60:i, 0])

x_test = np.array(x_test)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = load_model('D:/Projects/Google_Stock_Prediction/LSTM.h5')

predicted_stock = model.predict(x_test)
predicted_stock = scaler.inverse_transform(predicted_stock)

plt.plot(stock_price, color='red', label = 'Real Google Stock Price')
plt.plot(predicted_stock, color='blue', label= 'Predicted stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
