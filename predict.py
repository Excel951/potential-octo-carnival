import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#
# # Prediksi menggunakan model
# def forecast(model, series, time_step):
#     series = np.array(series).reshape(-1, 1)
#     data_scaled = scaler.transform(series)
#
#     dataset = tf.data.Dataset.from_tensor_slices(data_scaled)
#     dataset = dataset.window(size=time_step, shift=1, drop_remainder=True)
#     dataset = dataset.flat_map(lambda window: window.batch(time_step))
#     dataset = dataset.batch(1).prefetch(1)
#
#     forecast = model.predict(dataset)
#     forecast = scaler.inverse_transform(forecast)
#
#     return forecast
#
#
# # Prediksi untuk bulan berikutnya
# next_month_prediction = forecast(model, total_per_bulan['Total Kunjungan'].values, time_step)
# print("Forecast for next period:", next_month_prediction[-1])
