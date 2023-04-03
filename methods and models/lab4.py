import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from datetime import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data_split import train_test_data_setup


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./continuous dataset.csv', delimiter = ',', header = 0,infer_datetime_format=True)

data.head()

print(data.head().to_string())

target1, target2, target3 = data.loc[:45000,'nat_demand'], data.loc[:45000,'TQL_toc'], data.loc[:45000,'QV2M_toc']
pr1, pr2, pr3, pr4, pr5, pr6, pr7 = data['T2M_toc'],data['W2M_toc'],data['T2M_san'],data['QV2M_san'],data['W2M_san'],data['T2M_dav'],data['QV2M_dav']

at = {'target1': target1, 'target2': target2, 'target3': target3,
      'pr1:': pr1, 'pr2:': pr2, 'pr3:': pr3,
      'pr4:': pr4, 'pr5:': pr5, 'pr6:': pr6,
      'pr7:': pr7}
Data = pd.DataFrame(data = at)
targets = [target1, target2, target3]

fig, ax = plt.subplots(3)
sns.set(rc={'figure.figsize': (9, 6)})
ax[0].plot(target1)
ax[1].plot(target2)
ax[2].plot(target3)

plt.show()


def trendline(data, order):
    trend = np.polyfit(data.index.values, list(data), order)
    return np.poly1d(trend)(data.index.values)
i=2
for target in targets:
    trend = trendline(target, 3)
    plt.subplots(figsize=((10,8)))
    plt.plot(target)
    plt.plot(trend, linewidth=4)
    plt.show()
    i+=1


from statsmodels.tsa.stattools import adfuller
def stationar_test(target):
  test = adfuller(target)
  return(test[1])

print('For target1 p-value =',stationar_test(target1))
print('For target2 p-value =',stationar_test(target2))
print('For target3 p-value =',stationar_test(target3))

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
i=5
for target in targets:
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8))
    plot_acf(target, ax = ax1)
    plot_pacf(target, ax = ax2)
    plt.tight_layout()
    plt.show()
    i+=1

from statsmodels.tsa import stattools
N=72
for target in targets:
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8))
    ax1.plot(stattools.acovf(target,fft=False))
    ax2.plot(stattools.acovf(target, nlag=N, fft=False))
    plt.show()
    i+=1

sns.heatmap(Data.corr(method='pearson'), annot=True)
i+=1

w1, w2 = 24, 12
for target in targets:
    fig = plt.subplots(figsize=(10,8))
    rol_mean1 = target.rolling(w1).mean()
    rol_mean2 = target.rolling(w2).mean()
    plt.plot(target[5700:6000], label = 'Not filtered data')
    plt.plot(rol_mean2[5700:6000], linewidth=1, c = 'orange',label = f'First filtered, window={w2}')
    plt.plot(rol_mean1[5700:6000], linewidth=1, c = 'green', label = f'First filtered, window={w1}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    i+=1

from scipy import signal

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
for ax, target in zip(axes,targets):
    f, pxx = signal.welch(target, fs = 1000, nfft = 500, nperseg = 200)
    rol_mean1 = target.rolling(w1).mean()
    rol_mean2 = target.rolling(w2).mean()
    ff1, pxx_f1 = signal.welch(rol_mean1.dropna(), fs = 1000, nfft=500, nperseg=200)
    ff2, pxx_f2 = signal.welch(rol_mean2.dropna(), fs = 1000, nfft=500, nperseg=200)
    ax.plot(f, pxx, linewidth=3, label='True series')
    ax.plot(ff1, pxx_f1, linewidth=3, label='First filtered series')
    ax.plot(ff2, pxx_f2, linewidth=3, label='Second filtered series')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]') #Power spectral density
    ax.legend()
fig.tight_layout()
plt.show()
i+=1

discover = auto_arima(targets[0], stepwise = False, seasonal = False, trace = True)
discover.summary()

test, train = train_test_split(targets[0], test_size=0.9995, shuffle = False, random_state=42)
print(len(train))
print(len(test))

model = ARIMA(train, order=(2,1,3))
fitted = model.fit()


residuals = pd.DataFrame(fitted.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.tight_layout()
plt.show()
i+=1

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8))
plot_acf(residuals, ax = ax1)
plot_pacf(residuals, ax = ax2)
fig.savefig(f'./plot{i}')
plt.show()
i+=1

forecast_test = fitted.forecast(len(test))
fc = pd.Series([None]*len(train) + list(forecast_test))

d = {'target': targets[0], 'pred': fc}
df = pd.DataFrame(d)

df.plot()
plt.xlim(44950,45100)
plt.show()

mae = mean_absolute_error(test, forecast_test)
mape = mean_absolute_percentage_error(test, forecast_test)
rmse = np.sqrt(mean_squared_error(test, forecast_test))

print(f'mae: {mae}')
print(f'mape: {mape}')
print(f'rmse: {rmse}')


def wrap_into_input(forecast_length, feature_time_series, target_time_series):
    """ Convert data for FEDOT framework """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    input_data = InputData(idx=np.arange(0, len(target_time_series)),
                           features=feature_time_series, target=target_time_series,
                           task=task, data_type=DataTypesEnum.ts)

    return input_data


forecast_length = 200

ts_1 = wrap_into_input(forecast_length=forecast_length,
                       feature_time_series=np.array(data['nat_demand']),
                       target_time_series=np.array(data['nat_demand']))

train_ts_1, test_ts_1 = train_test_data_setup(ts_1)

ts_2 = wrap_into_input(forecast_length=forecast_length,
                       feature_time_series=np.array(data['T2M_toc']),
                       target_time_series=np.array(data['nat_demand']))

train_ts_2, test_ts_2 = train_test_data_setup(ts_2)

dataset = MultiModalData({
    'data_source_ts/nat_demand': ts_1,
    'data_source_ts/T2M_toc': ts_2
})

def simple_linear_pipeline():
    """ Pipeline lagged -> ridge """
    lagged_node = PrimaryNode('lagged')
    ridge_node = SecondaryNode('ridge', nodes_from=[lagged_node])
    return Pipeline(ridge_node)

simple_pipeline = simple_linear_pipeline()
simple_pipeline.show()
simple_pipeline.fit(train_ts_1)
forecast_uni = simple_pipeline.predict(test_ts_1)

train_length = len(data['nat_demand']) - forecast_length

plt.plot(data['nat_demand'], label='Actual time series')
plt.plot(np.arange(train_length, train_length + forecast_length),
         np.ravel(forecast_uni.predict), label='Forecast')
plt.xlim(train_length - 100, len(data['nat_demand']) + 10)
plt.legend()
plt.savefig('./plot_uni.png')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
MAE = mean_absolute_error(test_ts_1.target, np.ravel(forecast_uni.predict))
MAPE = mean_absolute_percentage_error(test_ts_1.target, np.ravel(forecast_uni.predict))
MSE = mean_squared_error(test_ts_1.target, np.ravel(forecast_uni.predict))
print(f'MAE metric value: {MAE:.2f} \n', f'MAPE metric value: {MAPE:.2f} \n', f'MAE metric value: {MSE:.2f} \n')

rol_mean2 = target.rolling(w2).mean()
rol = pd.DataFrame( data={'rol_mean2': rol_mean2.dropna().reset_index(drop=True)})

test, train = train_test_split(rol, test_size=0.9995, shuffle = False, random_state=42)
model = ARIMA(train, order=(2,1,3))
fitted = model.fit()


residuals = pd.DataFrame(fitted.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8))
plot_acf(residuals, ax = ax1)
plot_pacf(residuals, ax = ax2)
plt.tight_layout()
plt.show()

forecast_test = fitted.forecast(len(test))
fc = pd.Series([None]*len(train) + list(forecast_test))

d = {'target': rol['rol_mean2'], 'pred': fc}
df = pd.DataFrame(d)

df.plot()
plt.xlim(44950,45100)
plt.show()

mae = mean_absolute_error(test, forecast_test)
mape = mean_absolute_percentage_error(test, forecast_test)
rmse = np.sqrt(mean_squared_error(test, forecast_test))

print(f'mae: {mae}')
print(f'mape: {mape}')
print(f'rmse: {rmse}')
#seasonal_decompose(data.loc[:800,'nat_demand'], model='additive', period = 12).plot()


def create_multisource_pipeline():
    """ Generate pipeline with several data sources """
    node_source_1 = PrimaryNode('data_source_ts/nat_demand')
    node_source_2 = PrimaryNode('data_source_ts/T2M_toc')

    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_source_1])
    node_lagged_1.custom_params = {'window_size': 150}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source_2])

    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('linear', nodes_from=[node_ridge, node_lasso])
    pipeline = Pipeline(node_final)
    return pipeline





pipeline = create_multisource_pipeline()
pipeline.show()

pipeline.fit(train_ts_2)
forecast_multi = pipeline.predict(test_ts_2)

train_length = len(data['nat_demand']) - forecast_length

plt.plot(data['nat_demand'], label='Actual time series')
plt.plot(np.arange(train_length, train_length + forecast_length),
         np.ravel(forecast_multi.predict), label='Forecast')
plt.xlim(train_length - 100, len(data['nat_demand']) + 10)
plt.legend()
plt.savefig('./plot_multi.png')
plt.show()

MAE = mean_absolute_error(test_ts_2.target, np.ravel(forecast_multi.predict))
MAPE = mean_absolute_percentage_error(test_ts_2.target, np.ravel(forecast_multi.predict))
MSE = mean_squared_error(test_ts_2.target, np.ravel(forecast_multi.predict))
print(f'MAE metric value: {MAE:.2f} \n', f'MAPE metric value: {MAPE:.2f} \n', f'MAE metric value: {MSE:.2f} \n')




