# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:13:19 2020

@author: Steven
"""

from keras.models import Sequential
import pandas as pd
import numpy as np


import mysql.connector
from mysql.connector import Error
import math
from keras.layers.core import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime
import matplotlib.pyplot as plt
from datetime import datetime

try:
    connection = mysql.connector.connect(host='localhost',database='db_pibc_olap',user='root',password='')
    sql_select_Query = "select * from fact_harga where sk_rice_type=11 AND SK_MARKET=0 AND sk_date>=20160101 and sk_date<=20200131 ORDER BY sk_date"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    print("Total jumlah baris data aktual: ", cursor.rowcount)
    
    harga=[]
    str_tanggal=[]
    int_tanggal=[]
    
    print("\nCetak setiap record")
    for row in records:
        kolom_harga=row[4]
        date=row[1]
        tanggal=str(date)[:10]
        harga.append(kolom_harga)
        int_tanggal.append(date)
        str_tanggal.append(tanggal)
        #print("SK Date = ", row[1], )
        #print("Harga = ", row[4])


except Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if (connection.is_connected()):
        connection.close()
        cursor.close()
        print("MySQL connection is closed")
        
        
#SQL data 01012016 s.d 31072019

import mysql.connector
from mysql.connector import Error


try:
    connection = mysql.connector.connect(host='localhost',database='db_pibc_olap',user='root',password='')
    sql_select_Query = "select * from fact_harga where sk_rice_type=11 AND SK_MARKET=0 AND sk_date>=20160101 and sk_date<=20190731 ORDER BY sk_date"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    print("Total baris data train: ", cursor.rowcount)
    
    harga_percobaan=[]
    str_tanggal_percobaan=[]
 
    
    print("\nCetak setiap record")
    for row in records:
        list_harga_percobaan=row[4]
        date_percobaan=row[1]
        tanggal_percobaan=str(date_percobaan)[:10]
        harga_percobaan.append(list_harga_percobaan)
        str_tanggal_percobaan.append(tanggal_percobaan)
        #print("SK Date = ", row[1], )
        #print("Harga = ", row[4])


except Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if (connection.is_connected()):
        connection.close()
        cursor.close()
        print("MySQL connection is closed")

print(harga[:10])
print(str_tanggal[-10:])



#normalisasi dataset full actual 2016-2020

scaler = MinMaxScaler(feature_range = (0, 1))
array_harga = np.vstack(harga)
dataset = scaler.fit_transform(array_harga)
print(dataset[0:5])
len(dataset)

scaler = MinMaxScaler(feature_range = (0, 1))
array_data_percobaan = np.vstack(harga_percobaan)
dataset_percobaan = scaler.fit_transform(array_data_percobaan)
print(dataset_percobaan[0:5])
len(dataset_percobaan)


train_size = int(len(dataset_percobaan))
test_size = int(len(dataset) - train_size)


#pembagian dan inisialisasi data pada variable tes,dan train berdasarkan ukuran/size yang telah ditetapkan diatas

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Jumlah data train dan test (training set, test set): " + str((len(train), len(test))))

#pembagian data tes X, train X dan tes Y,train Y

def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset)):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i, 0])
    return(np.array(data_X), np.array(data_Y))

# Create test and training sets for one-step-ahead regression.
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)
print("Original train Y shape:")
print(train_Y.shape)



# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
#model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='relu'))


# we'll use mse for the loss, and RMSprop as the optimizer
print("Compile proses")
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print("Training...")
model.fit(train_X,train_Y, epochs=50, batch_size=16, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(test_X, verbose=0)


def predict_and_score(model, X, Y):
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

rmse_train, train_predict = predict_and_score(model, train_X, train_Y)
rmse_test, forecast = predict_and_score(model, test_X, test_Y)

print("\nAktual (5 awal)= ",harga_percobaan[:5])
print("Train Predict = ",train_predict[:5])
print("\nAktual (10 terakhir)= ",harga_percobaan[-10:])
print("Forecast Predict (10 terakhir)= ",forecast[-10:])
print("Forecast Shape = ",forecast.shape)
print("\n\nTraining data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

# Proses pembuatan dataframe dan inisialisasi tanggal pada variabel x_values

df_data=pd.DataFrame({"date":str_tanggal, "harga":harga}) 


#year = [datetime.strptime(date, '%Y') for date in str_tanggal]
print(df_data[-11:])
date = pd.Index(df_data["date"])
print(date)

x_values = [datetime.strptime(d,"%Y%m%d").date() for d in date]
print(x_values[-10:])

df_harga_real = pd.DataFrame(harga)

train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# MEmbuat plot dari data test/prediksi
forecast_predict_plot = np.empty_like(dataset)
forecast_predict_plot[:, :] = np.nan
forecast_predict_plot[len(train_predict) + (window_size * 2) - 3 :len(dataset) - 1, :] = forecast

plt.figure(figsize = (15, 5))

plt.plot(x_values,forecast_predict_plot, label = "Forecast/Data Test", color="red")
plt.plot(x_values,train_predict_plot,label = "Data Train", linewidth=4)
plt.plot(x_values,df_harga_real,label = "Harga beras", color="yellow", linestyle='dashed')


plt.xlabel("WAKTU")
plt.ylabel("HARGA BERAS")
plt.title("Comparison true vs. predicted")
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.legend()
plt.show()