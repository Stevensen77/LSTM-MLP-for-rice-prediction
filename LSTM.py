
'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

======================= LSTM =============================
'''


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from mysql.connector import Error
import mysql.connector
from tensorflow.keras.models import load_model

## mengambil data
try:
    connection = mysql.connector.connect(host='localhost',database='lstm_beras',user='root',password='')
    sql_select_Query = "select * from lstm_data_beras"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    print("Total jumlah baris data aktual: ", cursor.rowcount)

    harga=[]
    str_tanggal=[]
    int_tanggal=[]

    print("\nCetak setiap record")
    for row in records:
        kolom_harga=row[2]
        date=row[1]
        tanggal=str(date)[:10]
        harga.append(kolom_harga)
        int_tanggal.append(date)
        str_tanggal.append(tanggal)
    

except Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if (connection.is_connected()):
        connection.close()
        cursor.close()
        print("MySQL connection is closed")

print("\n\nHarga aktual (10 pertama) : \n",harga[:10])

print("\n\nTanggal aktual (10 terakhir) : \n",str_tanggal[-10:])

print("panjang tanggal :",len(str_tanggal))

#normalisasi dataset full actual 2016-2020

scaler = MinMaxScaler(feature_range = (0, 1))
array_harga = np.vstack(harga)
dataset = scaler.fit_transform(array_harga)
print("\n\nMinmaxscaler isi : ",dataset[0:7])


#pembagian data tes dan train

train_size = int(len(dataset)*0.8)
test_size = int(len(dataset) - train_size)


print("\nPanjang train: ", train_size)
print("Panjang test : ", test_size)

array_tanggal = np.array(str_tanggal)


tanggal_train= str_tanggal[0:train_size]
tanggal_test=str_tanggal[train_size:len(str_tanggal)]


print("\n\nTanggal awal train :",tanggal_train[0:5])
print("\n\nTanggal akhir train :",tanggal_train[-5:])
print("\n\nTanggal awal test :",tanggal_test[0:5])
print("\n\nTanggal akhir test :",tanggal_test[-5:])


#pembagian dan inisialisasi data pada variable tes,dan train berdasarkan ukuran/size yang telah ditetapkan diatas

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("\n\n Komposisi data train dan test (training set, test set) = " + str((len(train), len(test))))



#pembagian data tes X, train X dan tes Y,train Y

def create_dataset(dataset, look_back = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_X.append(a)
        data_Y.append(dataset[i + look_back, 0])

    return(np.array(data_X), np.array(data_Y))


look_back = 1

	
# reshape into X=t and Y=t+1  (t = timestep)
train_X, train_Y = create_dataset(train, look_back)
test_X, test_Y = create_dataset(test, look_back)



print("\nOriginal train_X shape:")
print(train_X.shape)

print("\nOriginal train_Y shape:")
print(train_Y.shape)


print("\nOriginal test_X shape:")
print(test_X.shape)

print("\nOriginal test_y shape:")
print(test_Y.shape)



# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("\n\nNew training data shape:")
print(train_X.shape)
print("New test_X shape:")
print(test_X.shape,"\n")


#Fungsi pembangunan dan training model LSTM
timestep=20

def fit_model(train_X, train_Y, look_back = 1):
    model = Sequential()
    
    model.add(LSTM(128,input_shape = (timestep, look_back)))
    
    #model.add(LSTM(128,return_sequences=True))
    #model.add(LSTM(128))
    model.add(Dense(1))
    model.compile(loss      = "mean_squared_error",
                  optimizer = "adam")
    model.fit(train_X,
              train_Y,
              epochs     = 100,
              batch_size = 32,
              verbose = 2)

    return(model)


# Memanggil fungsi latih

model1 = fit_model(train_X, train_Y, look_back)

#  ----------- Bobot ---------

#print("\n\n ISI weight (5 awal ): \n", model1.layers[0].trainable_weights)

units = int(int(model1.layers[0].trainable_weights[0].shape[1])/4)
print("\n\nNo units: ", units)

W = model1.layers[0].get_weights()[0]
U = model1.layers[0].get_weights()[1]
b = model1.layers[0].get_weights()[2]

W_i = W[:, :units]
W_f = W[:, units: units * 2]
W_c = W[:, units * 2: units * 3]
W_o = W[:, units * 3:]

U_i = U[:, :units]
U_f = U[:, units: units * 2]
U_c = U[:, units * 2: units * 3]
U_o = U[:, units * 3:]

b_i = b[:units]
b_f = b[units: units * 2]
b_c = b[units * 2: units * 3]
b_o = b[units * 3:]

'''
print("\n\n Bobot W (5 awal)= ",W[:5])
print("\n\n Bobot U  = ",U)
print("\n\n Bias b = ",b)
print("\n\n W_1 = ",W_i)
print("\n\n W_f = ",W_f)
print("\n\n W_c = ",W_c)
'''

#print("\n\n Model Summary : \n", model1.summary())


#menyimpan model
model1.save("model_lstm.h5")
print("\n\nModel Berhasil TERSIMPAN\n\n")


#model1 = load_model('C:\django\django_lstm\lstm\model_lstm.h5')
    

# Membuat nilai forecast dan nilai RMSE

print("\n\nisi train X itu sebelum predict (10 pertama) = ",train_X[0:10])
print("\n\nisi test_x sebelum predict (10 pertama)=",test_X[:10])
print(type(test_X))

'''
def predict_and_score(model, X, Y):
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))

    return(score, pred)
'''

# make predictions
trainPredict = model1.predict(train_X)
testPredict = model1.predict(test_X)

# invert predictions
train_predict = scaler.inverse_transform(trainPredict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(testPredict)
test_Y = scaler.inverse_transform([test_Y])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_Y[0], train_predict[:,0]))
print('\nTrain Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_Y[0], test_predict[:,0]))
print('\nTest Score: %.2f RMSE' % (testScore))

'''
rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, forecast = predict_and_score(model1, test_X, test_Y)
'''

print("\n\nTrain Predict (10 pertama)= ",train_predict[:10])
print("\n\nFaktual (10 pertama)= ",harga[:10])
print("\n\nPredict test ( last 10 )= ",test_predict[-10:])
print("\n\nFaktual (10 terakhir)= ",harga[-10:])

'''
print("\n\nTraining data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)
'''

# Proses pembuatan dataframe dan inisialisasi tanggal pada variabel x_values

df_data=pd.DataFrame({"date":str_tanggal, "harga":harga})


#year = [datetime.strptime(date, '%Y') for date in str_tanggal]
print("\n\nData Frame Aktual (10 terakhir) \n",df_data[-11:])
date = pd.Index(df_data["date"])


x_values = [datetime.strptime(d,"%Y%m%d").date() for d in date]
print("\n\nFormat tanggal : ",x_values[-10:])


# penggabungan antara nilai train dan prediksi untuk keperluan plotting
prediksi=np.concatenate((train_predict, test_predict), axis=0)
print(type(prediksi))


df_harga_real = pd.DataFrame(harga)

print("\n\nData Harga Aktual (menampilkan 10 row pertama) \n",df_harga_real[0:10])
print("\nData Harga Aktual (menampilkan 10 row terakhir) \n",df_harga_real[-10:])



# Membuat plot dari data train.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
print("\n\nTrain predict shape :",train_predict.shape)
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
print("Train predict plot shape :",train_predict_plot.shape)

# MEmbuat plot dari data test/prediksi
forecast_predict_plot = np.empty_like(dataset)
forecast_predict_plot[:, :] = np.nan
forecast_predict_plot[len(train_predict) + (look_back * 2) + 1 :len(dataset) - 1, :] = test_predict
print("Test predict plot shape :",forecast_predict_plot.shape)



plt.figure(figsize = (15, 5))

plt.plot(x_values,forecast_predict_plot, label = "Forecast/Data Test", color="red", linewidth=4)
plt.plot(x_values,train_predict_plot,label = "Data Train", color="yellow", linewidth=4)
plt.plot(x_values,df_harga_real,label = "Harga beras", color="green", linestyle='dashed',linewidth=2)


plt.xlabel("WAKTU")
plt.ylabel("HARGA BERAS")
plt.title("Perbandingan harga aktual vs. prediksi")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.show()


# ============ PEMROSESAN UNTUK DATA GRAFIK DJANGO ===============
print("\n\n =======================================================================")
print("\n\ntipe tanggal",type(str_tanggal))
int_tanggal = list(map(int, str_tanggal))
print("\n\nTanggal awal integer :",int_tanggal[:5])

list_tahun=[]
list_tahun_unik=[]
list_float_train=[]
list_float_test=[]
float_list_prediksi=[]

for a in str_tanggal:
    str_tahun = a[:4]
    list_tahun.append(int(str_tahun))

(unique) = np.unique(list_tahun, return_counts=False)
arr_tahun_unik = np.asarray((unique)).T

list_tahun_unik=arr_tahun_unik.tolist()


print("\n\nIsi tahun unik  :",list_tahun_unik)



list_prediksi=prediksi.tolist()
for i,a in enumerate(list_prediksi):
    str_prediksi = str(a)
    str_prediksi2= str_prediksi.replace("[", "").replace("]", "")
    float_prediksi=float(str_prediksi2[:8])
    float_list_prediksi.append(float_prediksi)
print("\n\nFloat Prediksi (menampilkan 10 row pertama) : \n ",float_list_prediksi[:10])
#print("\n\nPrediksi (menampilkan 10 row pertama) : \n ",prediksi[:10])
print("\n\nFloat Prediksi (menampilkan 10 row terakhir) : \n ",float_list_prediksi[-10:])
#print("Prediksi (menampilkan 10 row terakhir) : \n ",prediksi[-10:])

print("\n\n  +++++++  Panjang prediksi : ",len(float_list_prediksi))

for i,a in enumerate(train_predict):
    str_train = str(a)
    str_train= str_train.replace("[", "").replace("]", "")
    float_train=float(str_train[:8])
    list_float_train.append(float_train)

print("\n\n LIST Float Train (10 awal) : \n",list_float_train[:10])

print("\n\n +++++++ Panjang train : ",len(list_float_train))


for i,a in enumerate(test_predict):
    str_test = str(a)
    str_test= str_test.replace("[", "").replace("]", "")
    float_test=float(str_test[:8])
    list_float_test.append(float_test)

print("\n\n LIST Float Test (10 akhir) :\n",list_float_test[-10:])
print("\n\n  +++++++  Panjang test : ",len(list_float_test))



list_float_train=float_list_prediksi[:len(train_predict)]
list_float_test=float_list_prediksi[len(train_predict):]

print("\n\n  +++++++  Panjang Test untuk grafik : ",len(list_float_test))

list_2016=[]
list_2017=[]
list_2018=[]
list_2019=[]
list_2020=[]


for i,a in enumerate(int_tanggal):
    if(a==20160101):
        list_2016.append(harga[i])
    if(a==20160301):
        list_2016.append(harga[i])
    if(a==20160601):
        list_2016.append(harga[i])
    if(a==20160901):
        list_2016.append(harga[i])
    if(a==20161201):
        list_2016.append(harga[i])
    if(a==20170101):
        list_2017.append(harga[i])
    if(a==20170301):
        list_2017.append(harga[i])
    if(a==20170601):
        list_2017.append(harga[i])
    if(a==20170901):
        list_2017.append(harga[i])
    if(a==20171201):
        list_2017.append(harga[i])
    if(a==20180101):
        list_2018.append(harga[i])
    if(a==20180301):
        list_2018.append(harga[i])
    if(a==20180601):
        list_2018.append(harga[i])
    if(a==20180901):
        list_2018.append(harga[i])
    if(a==20181201):
        list_2018.append(harga[i])
    if(a==20190101):
        list_2019.append(harga[i])
    if(a==20190301):
        list_2019.append(harga[i])
    if(a==20190601):
        list_2019.append(harga[i])
    if(a==20190901):
        list_2019.append(harga[i])
    if(a==20191201):
        list_2019.append(harga[i])
    if(a==20200101):
        list_2020.append(harga[i])
    if(a==20200301):
        list_2020.append(harga[i])




print("\n\n ISI list harga pilihan 2016: \n", list_2016)

print("\n\n ISI list harga pilihan 2017: \n", list_2017)

print("\n\n ISI list harga pilihan 2018: \n", list_2018)

print("\n\n ISI list harga pilihan 2019: \n", list_2019)

print("\n\n ISI list harga pilihan 2020: \n", list_2020)


head_harga_aktual=harga[:10]
tail_harga_aktual=harga[-10:]
head_tanggal=int_tanggal[:10]
tail_tanggal=int_tanggal[-10:]



head_harga_prediksi=float_list_prediksi[:10]

df_gabungan_head_tgl_hrg_prediksi = pd.DataFrame
({
        "Tanggal": head_tanggal,
        "Harga": head_harga_aktual,
        "Prediksi":head_harga_prediksi,
})

print("isi head prediksi",head_harga_prediksi)





print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n +++++++++++++++++++++++++++++ TAHAPAN PREDIKSI FUTURE ++++++++++++++++++++++++++ \n")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

tanggal_new = [datetime.strptime(d,"%Y%m%d").date().strftime('%Y-%m-%d') for d in date]

df_new=pd.DataFrame({"date":tanggal_new, "harga":harga})


print("\n df baru",df_new[-10:])


df_new['date']=pd.to_datetime(df_new.date)


df = df_new.set_index("date")

train = df
scaler.fit(train)
train = scaler.transform(train)
n_input = 100
jarak_index=int(n_input+1)
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=64)
model1.fit_generator(generator,epochs=100)

model1.save("model_lstm_till_future.h5")
print("\n\nModel FUTURE Berhasil TERSIMPAN\n\n")


pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):
    pred_list.append(model1.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

#print("\n\nISI pred list FUTURE:",pred_list)

add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0,jarak_index)  ]
print("\n\n Tipe add date : ",type(add_dates))
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])


print("\n + PREDIKSI FUTURE (10 awal):\n",df_predict[:10])

df_proj = pd.concat([df,df_predict], axis=1)

print("\n + PREDIKSI FUTURE (10 akhir):\n",df_proj[-10:])

plt.figure(figsize=(15, 5))
plt.xlabel("WAKTU")
plt.ylabel("HARGA BERAS")
plt.plot(df_proj.index, df_proj['harga'], label = "Harga Real")
plt.plot(df_proj.index, df_proj['Prediction'], color='r', label = "Forecast")
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()



print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++======-========+++++++")




