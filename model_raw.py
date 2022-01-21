"""# Data Loading"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
import zipfile
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

url = 'https://query1.finance.yahoo.com/v7/finance/download/GOLD.AX?period1=1546300800&period2=1641772800&interval=1d&events=history&includeAdjustedClose=true'
df = pd.read_csv(url)
df

date = df['Date'].values
close  = df['Adj Close'].values

plt.plot(date, close)
plt.title('Gold Price 2019-2022', fontsize=20);

print(df['Adj Close'][0])
print(df['Adj Close'][200])
print(df['Adj Close'][400])
print(df['Adj Close'][689])  # df - 10%
print(df['Adj Close'][650])  # df - 15%
print(df['Adj Close'][612])  # df - 20%
print(df['Adj Close'][700])
print(df['Adj Close'][766])

df.describe()

close_median = df['Adj Close'].median()
vol_median = df['Volume'].median()
open_median = df['Open'].median()
high_median = df['High'].median()
low_median = df['Low'].median()

print(close_median)
print(vol_median)
print(open_median)
print(high_median)
print(low_median)

"""# Mendeteksi Missing Value"""

df.info()

"""# Menangani Outliers"""

sns.boxplot(x=df['Adj Close'])

sns.boxplot(x=df['Volume'])

df['Adj Close'].hist()

df['Volume'].hist()

df['Open'].hist()

df['High'].hist()

df['Low'].hist()

print('skewness value of Adj Close: ',df['Adj Close'].skew())
print('skewness value of Volume: ',df['Volume'].skew())
print('skewness value of Open: ',df['Open'].skew())
print('skewness value of High: ',df['High'].skew())
print('skewness value of Low: ',df['Low'].skew())

Q1 = df['Volume'].quantile(0.25)
Q3 = df['Volume'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5

vol_outliers = df[(df['Volume'] < Q1 - whisker_width*IQR) | (df['Volume'] > Q3 + whisker_width*IQR)]
vol_values=pd.DataFrame(vol_outliers['Volume'])
vol_values = vol_values.values
vol_median = df['Volume'].median()

df['Volume'] = df['Volume'].replace(to_replace=vol_values, value=vol_median)
df

sns.pairplot(df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title('Correlation Matrix untuk Fitur Numerik', size=20)

"""# Data Preparation"""

x=pd.DataFrame(df[['Open']])
y=pd.DataFrame(df['Adj Close'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)

x_train

y_train

numerical_features = ['Open']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()

x_train

y_train

x_test

y_test

"""# Model Development"""

# dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'KNNTune1', 'RandomForest', 'RandomForestTune1', 'Boosting', 'BoostingTune1', 'NN', 'NNTune1'])
models

"""## K-Nearest Neighbor Models"""

knn = KNeighborsRegressor(n_neighbors=10, metric='euclidean')
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_train)

knn_hypt = KNeighborsRegressor()

knn_params = [{'n_neighbors': (range(10, 50)), 'algorithm': ('ball_tree', 'kd_tree', 'brute'), 'metric': ('minkowski', 'euclidean')}]
knngs = GridSearchCV(knn_hypt, knn_params, cv = 15, scoring='neg_mean_squared_error')

knngs.fit(x_train, y_train)

print(knngs.best_params_)
print(knngs.best_score_)

knn_hypt = KNeighborsRegressor(algorithm='brute', metric='minkowski', n_neighbors=10)

knn_hypt.fit(x_train, y_train)
y_pred_knn_hypt = knn_hypt.predict(x_train)

"""## Random Forest Models"""

RF = RandomForestRegressor(n_estimators=80, max_depth=20, random_state=None, n_jobs=-1)
RF.fit(x_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)

RF_hypt = RandomForestRegressor()

rf_params = [{'n_estimators': (range(100, 135)), 'max_depth': (None, range(1, 30))}]
rfgs = GridSearchCV(RF_hypt, rf_params, cv = 5, scoring='neg_mean_squared_error')

rfgs.fit(x_train, y_train)

print(rfgs.best_params_)
print(rfgs.best_score_)

RF_hypt = RandomForestRegressor(max_depth=None, n_estimators=134)

RF_hypt.fit(x_train, y_train)
models.loc['train_mse','RandomForestTune1'] = mean_squared_error(y_pred=RF_hypt.predict(x_train), y_true=y_train)

"""## Boosting Algorithm"""

boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05)                             
boosting.fit(x_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(x_train), y_true=y_train)

boosting_hypt = AdaBoostRegressor()

boost_params = [{'learning_rate': (0.5, 0.05), 'n_estimators': (range(60, 90))}]
btgs = GridSearchCV(boosting_hypt, boost_params, cv = 15, scoring='neg_mean_squared_error')

btgs.fit(x_train, y_train)

print(btgs.best_params_)
print(btgs.best_score_)

boosting_hypt = AdaBoostRegressor(learning_rate=0.5, n_estimators=88)

boosting_hypt.fit(x_train, y_train)
models.loc['train_mse','BoostingTune1'] = mean_squared_error(y_pred=boosting_hypt.predict(x_train), y_true=y_train)

"""## Neural Network"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

optimizer = tf.keras.optimizers.SGD(lr=1.0000e-04, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), 
              optimizer=optimizer, 
              metrics=['mse'])

history = model.fit(
    x_train.values, 
    y_train.values, 
    epochs=10, 
    batch_size=64,
    validation_split=0.2)

model_hypt = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

optimizer_hypt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model_hypt.compile(loss=tf.keras.losses.Huber(), 
              optimizer=optimizer_hypt, 
              metrics=['mse'])

history_hypt = model_hypt.fit(
    x_train.values, 
    y_train.values, 
    epochs=30, 
    batch_size=64,
    validation_split=0.2)

"""# Evaluation"""

x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])
x_test

x_test.describe()

x_train

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','KNNTune1','RF','RFTune1','Boosting','BoostingTune1','NN','NNTune1'])
model_dict = {'KNN':knn, 'KNNTune1':knn_hypt, 'RF':RF, 'RFTune1':RF_hypt, 'Boosting':boosting, 'BoostingTune1':boosting_hypt, 'NN':model, 'NNTune1':model_hypt}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3
 
mse

mse.drop(['NN', 'NNTune1'], axis=0, inplace=True)
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=True).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

mse.drop(['Boosting'], axis=0, inplace=True)
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=True).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = x_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pred_dict

raw = y.pct_change(periods=766, fill_method ='ffill').mean()
percentage_raw = raw*100
percentage = str(percentage_raw).split()

print('Rata-rata kenaikan harga emas dari periode January 2019-2022 adalah {0}%'.format(percentage[2]))

df

nilai_awal = df['Adj Close'][0]
nilai_kenaikan = nilai_awal * (percentage_raw / 100)
nilai_akhir = nilai_awal + nilai_kenaikan

print('nilai harga emas pada akhir dataset (row 766): {}'.format(nilai_akhir))

nilai_kenaikan

# rata-rata persentase kenaikan perhari
raw_h = y.pct_change(periods=1, fill_method ='ffill').mean()
raw_percentage_h = raw_h*100  # output yang dikeluarkan belum berupa persentase
percentage_h = str(raw_percentage_h).split()

print('Rata-rata kenaikan harga emas per hari adalah {0}%'.format(percentage_h[2]))
