import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dir2024 = '2024'
jan24 = pd.read_csv(
    os.path.join(dir2024, 'januari-banyaknya-kunjungan-baru-lama-di-puskesmas-menurut-poli-per-kecamatan.csv'),
    delimiter=';', encoding='iso-8859-1')
feb24 = pd.read_csv(
    os.path.join(dir2024, 'februari-banyaknya-kunjungan-baru-lama-di-puskesmas-menurut-poli-per-kecamatan.csv'),
    delimiter=';', encoding='iso-8859-1')
mar24 = pd.read_csv(
    os.path.join(dir2024, 'maret-banyaknya-kunjungan-baru-lama-di-puskesmas-menurut-poli-per-kecamatan.csv'),
    delimiter=';', encoding='iso-8859-1')
apr24 = pd.read_csv(
    os.path.join(dir2024, 'april-banyaknya-kunjungan-baru-lama-di-puskesmas-menurut-poli-per-kecamatan.csv'),
    delimiter=';', encoding='iso-8859-1')
mei24 = pd.read_csv(
    os.path.join(dir2024, 'mei-banyaknya-kunjungan-baru-lama-di-puskesmas-menurut-poli-per-kecamatan.csv'),
    delimiter=';', encoding='iso-8859-1')

jan23 = pd.read_csv('./2023/januari-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';',
                    encoding='iso-8859-1')
feb23 = pd.read_csv('./2023/februari-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';',
                    encoding='iso-8859-1')
mar23 = pd.read_csv('./2023/maret-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';', encoding='iso-8859-1')
apr23 = pd.read_csv('./2023/april-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';', encoding='iso-8859-1')
mei23 = pd.read_csv('./2023/mei-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';', encoding='iso-8859-1')
jun23 = pd.read_csv('./2023/juni-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';', encoding='iso-8859-1')
jul23 = pd.read_csv('./2023/juli-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';', encoding='iso-8859-1')
agu23 = pd.read_csv('./2023/agustus-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';',
                    encoding='iso-8859-1')
sep23 = pd.read_csv('./2023/september-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';',
                    encoding='iso-8859-1')
okt23 = pd.read_csv('./2023/oktober-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';',
                    encoding='iso-8859-1')
nov23 = pd.read_csv('./2023/november-jumlah-kunjungan-puskesmas-lama-baru-2023.csv', delimiter=';',
                    encoding='iso-8859-1')
des23 = pd.read_csv('./2023/desember-banyaknya-kunjungan-baru-lama-di-puskesmas-menurut-poli-per-kecamatan-1-1.csv',
                    delimiter=';', encoding='iso-8859-1')

jan22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-januari-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
feb22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-februari-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
mar22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-maret-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
apr22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-april-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
mei22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-mei-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
jun22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-juni-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
jul22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-juli-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
agu22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-bulan-agustus-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
sep22 = pd.read_csv('./2022/banyaknya-kunjungan-di-puskesmas-per-kecamatan-tahun-2022-september.csv', delimiter=';',
                    encoding='iso-8859-1')
okt22 = pd.read_csv('./2022/oktober-banyaknya-kunjungan-di-puskesmas-per-kecamatan-tahun-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
nov22 = pd.read_csv('./2022/november-banyaknya-kunjungan-di-puskesmas-per-kecamatan-tahun-2022.csv', delimiter=';',
                    encoding='iso-8859-1')
des22 = pd.read_csv('./2022/desember-banyaknya-kunjungan-di-puskesmas-per-kecamatan-tahun-2022.csv', delimiter=';',
                    encoding='iso-8859-1')

# Menambahkan kolom 'Periode' ke setiap DataFrame
jan24['Periode'] = 'January 2024'
feb24['Periode'] = 'February 2024'
mar24['Periode'] = 'March 2024'
apr24['Periode'] = 'April 2024'
mei24['Periode'] = 'May 2024'

jan23['Periode'] = 'January 2023'
feb23['Periode'] = 'February 2023'
mar23['Periode'] = 'March 2023'
apr23['Periode'] = 'April 2023'
mei23['Periode'] = 'May 2023'
jun23['Periode'] = 'June 2023'
jul23['Periode'] = 'July 2023'
agu23['Periode'] = 'August 2023'
sep23['Periode'] = 'September 2023'
okt23['Periode'] = 'October 2023'
nov23['Periode'] = 'November 2023'
des23['Periode'] = 'December 2023'

jan22['Periode'] = 'January 2022'
feb22['Periode'] = 'February 2022'
mar22['Periode'] = 'March 2022'
apr22['Periode'] = 'April 2022'
mei22['Periode'] = 'May 2022'
jun22['Periode'] = 'June 2022'
jul22['Periode'] = 'July 2022'
agu22['Periode'] = 'August 2022'
sep22['Periode'] = 'September 2022'
okt22['Periode'] = 'October 2022'
nov22['Periode'] = 'November 2022'
des22['Periode'] = 'December 2022'

data = pd.concat([
    jan22, feb22, mar22, apr22, mei22, jun22, jul22, agu22, sep22, okt22, nov22, des22, jan23, feb23, mar23, apr23,
    mei23, jun23, jul23, agu23, sep23, okt23, nov23, des23, jan24, feb24, mar24, apr24, mei24

])
# val_data = pd.concat([
#     mar24, apr24, mei24
# ])

# Filter data untuk Puskesmas A (contoh)
puskesmas_name = 'Puskesmas Dukuh Kupang'
data_puskesmas_a = data[data['Nama Puskesmas'] == puskesmas_name]

total_per_bulan = data_puskesmas_a.groupby('Periode')['Total Kunjungan'].sum().reset_index()
total_per_bulan['Periode'] = pd.to_datetime(total_per_bulan['Periode'], format='%B %Y')
total_per_bulan = total_per_bulan.sort_values('Periode')
# print('Dataset', total_per_bulan)
print(total_per_bulan.describe())

# plt.figure(figsize=(12, 6))
# plt.plot(total_per_bulan['Periode'], total_per_bulan['Total Kunjungan'], marker='o')
#
# plt.title(f'Total Kunjungan Puskesmas {puskesmas_name} per Bulan')
# plt.xlabel('Periode')
# plt.ylabel('Total Kunjungan')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.ylim(0)  # Memastikan sumbu Y dimulai dari 0
# plt.tight_layout()
# plt.show()

# val_total_per_bulan = val_data.groupby(['Periode', 'Nama Puskesmas'])['Total Kunjungan'].sum().reset_index()
# val_total_per_bulan['Periode'] = pd.to_datetime(val_total_per_bulan['Periode'], format='%B %Y')
# val_total_per_bulan = val_total_per_bulan.sort_values('Periode')
# # print('Val', val_total_per_bulan)
# print(val_total_per_bulan.describe())

# Normalisasi data Total Kunjungan
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(total_per_bulan['Total Kunjungan'].values.reshape(-1, 1))

# Menentukan parameter untuk jendela waktu
time_step = 6  # Jumlah bulan yang digunakan sebagai input untuk memprediksi bulan berikutnya
BATCH_SIZE = 3
N_PAST = 6  # Jumlah bulan sebelumnya yang digunakan untuk prediksi
N_FUTURE = 2  # Prediksi hanya satu bulan ke depan
SHIFT = 1  # Perpindahan jendela waktu


# Fungsi untuk membuat dataset dengan jendela waktu
def windowed_dataset(series, batch_size, n_past=3, n_future=1, shift=1):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_past + n_future))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda window: (window[:-n_future], window[-n_future:, :1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# Buat dataset menggunakan windowed_dataset function
dataset = windowed_dataset(data_scaled, BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)

# Bangun model LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(N_PAST, 1)),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Latih model
histori = model.fit(dataset, epochs=20)


# Prediksi untuk bulan berikutnya
def forecast(model, series, time_step):
    series = np.array(series).reshape(-1, 1)
    data_scaled = scaler.transform(series)

    dataset = tf.data.Dataset.from_tensor_slices(data_scaled)
    dataset = dataset.window(size=time_step, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(time_step))
    dataset = dataset.batch(1).prefetch(1)

    forecast = model.predict(dataset)
    forecast = scaler.inverse_transform(forecast)

    return forecast


# Contoh prediksi untuk bulan berikutnya
last_n_values = data_puskesmas_a[data_puskesmas_a['Periode'] == data_puskesmas_a['Periode'].iloc[-1]][
                    'Total Kunjungan'].values[-N_PAST:]
next_month_prediction = forecast(model, last_n_values, N_PAST)
prediction = round(next_month_prediction[0][0])
print("Forecast for next period:", prediction)

# total_per_bulan.tail[1] = ['2024-06-01', int(prediction)]
# print(total_per_bulan)
#
# plt.figure(figsize=(12, 6))
# plt.plot(total_per_bulan['Periode'], total_per_bulan['Total Kunjungan'], marker='o')
#
# plt.title(f'Total Kunjungan Puskesmas {puskesmas_name} per Bulan')
# plt.xlabel('Periode')
# plt.ylabel('Total Kunjungan')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.ylim(0)  # Memastikan sumbu Y dimulai dari 0
# plt.tight_layout()
# plt.show()
