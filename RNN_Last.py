# -*- coding: utf-8 -*-
"""RNN/LSTM Electrical Load prediction"""
#%% Packages
import keras # You can safely ignore any warnings on importing this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% Import load data in 15 min intervall
df = pd.read_csv('data_prepared.csv', index_col=0, sep=',', parse_dates=True, 
                 infer_datetime_format=True, dayfirst=True)
df.index.freq = '15T'

# Reduce data amount to from 2 years to 10 d (to improve train speed)
data_days = 10
df = df[-data_days*96:]

# Extract the relevant data "Hges"
df_bck = df
df = df['Hges']
df = df.to_frame() # Conversion from series to df necessary

#%% Train Test Split
train = df.iloc[:-96]
test = df.iloc[-96:]

#%% Normalize data (0 to 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test) 

#%% Time Series Generator
from keras.preprocessing.sequence import TimeseriesGenerator
# define generator
n_input = 96 # calculate with 96 time stamps
batch_size = 1
n_features = 1 # univariate time series (without additional features)
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=batch_size)

#%% Create LSTM-model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(None, n_features)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(n_features, return_sequences=True)) 
model.compile(loss='mse', optimizer='adam')

print(model.summary()) # Model summary

#%% Train Model
num_epochs = 1000
model.fit_generator(train_generator, epochs=num_epochs)
# Plot the loss per epoch (Alternative use TensorBoard)
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

#%% Create prediction
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0] 
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
# Invert normalization
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions

#%% Compare
from custom_fun import calculate_stats
title = "LSTM Neurons, " + str(num_epochs) + " Epochs)"
# np.nan bei AIC Berechnung, da ich die Anzahl der Parameter nicht kenne hier
calculate_stats(title, test['Hges'], test['Predictions'], len(train), np.nan) 
test.plot()

#%% Save Model
model.save('LSTM_LoadPrediction.h5')
