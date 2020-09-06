# -*- coding: utf-8 -*-
"""RNN/LSTM Lastprognose

Mega Hinweis: WENN ICH NUR 24h in die Zukunft blicken will, brauche ich keine 3 Jahre an Daten!
Es reicht vielleicht der letzte Monat! Sensitivitätsanalyse machen!

Schauen ob ich einen kleinen Zeitraum finde, in dem alles vorkommt (nzht, zht, at, ...)

Weniger Daten sparen Trainingszeit.

Mit df.info() zähle ich direkt Datenlücken nan im Datensatz.
"""
#%% Vorbereitung
import keras # You can safely ignore any warnings on importing this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data_prepared.csv', index_col=0, sep=',', parse_dates=True, 
                 infer_datetime_format=True, dayfirst=True)
df.index.freq = '15T'

# Datenmenge verkleinern auf Tagebasis
data_days = 10
df = df[-data_days*96:]

# Hges vom Rest trennen
df_bck = df
df = df['Hges']
df = df.to_frame() # Umwandlung von ser zu df nötig

#%% Train Test Split
train = df.iloc[:-96]
test = df.iloc[-96:]

#%% Skalierung (0 bis 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test) 

#%% Time Series Generator
from keras.preprocessing.sequence import TimeseriesGenerator
# define generator
n_input = 96 # Ich gebe dir die letzten 96 Datenpunkte (=1 Tag) und du sagst mir wie die nächsten
# 15 min aussehen (97. Datenpunkt)
# Mit n_input kann man ausprobieren
batch_size = 1
n_features = 1 # Wieviele Datenspalten man hat steht hier (nur Hges)
# 2 x scaled_train, weil die Datenreihe und die target Werte eingegeben werden müssen
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, 
                                      batch_size=batch_size)

#%% KNN Modell initiieren
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

"""
# Modell initiieren
model = Sequential()
# LST Layer einbauen mit x LSTM Neuronen

model.add(LSTM(128, activation='relu', input_shape=(n_input, n_features), 
               return_sequences=True))

model.add(LSTM(128, return_sequences=False))
model.add(LSTM(64, return_sequences=False))

# Zu einem Output Wert zusammenführen
model.add(Dense(1))
# Compile
model.compile(optimizer='adam', loss='mse')
"""

# Summary des Modells:
print(model.summary())


#%% Modell trainieren
num_epochs = 1000
model.fit_generator(train_generator, epochs=num_epochs)
# Fehler in Abhängigkeit der Epochen plotten
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

#%% Prognose erstellen
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0] 
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
# Scaling Prozess invertieren
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions

#%% Vergleichen
from custom_fun import calculate_stats
title = "LSTM Neurons, " + str(num_epochs) + " Epochs)"
# np.nan bei AIC Berechnung, da ich die Anzahl der Parameter nicht kenne hier
calculate_stats(title, test['Hges'], test['Predictions'], len(train), np.nan) 
test.plot()

#%% Modell speichern und laden
# Damit man dieses Modell wieder aufrufen kann
model.save('RNN_Lastmodell.h5')