import numpy as np
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore")

pd.set_option("display.max.columns", None)
url = "C:/TimeSeries/final project/AAPL (1).csv"
data = pd.read_csv(url)
print(data)

print(data.columns)

#Data Cleaning
print('\n')
print(data.describe(include='all'))

#Data types
print(f'\n Data types:{data.dtypes}')

#Checking Null values
print('\nFinding NaN values\n', data.isnull().sum())

#Check wether the data is Equally sampled or not
data['Date'] = pd.to_datetime(data['Date'])
time_intervals = data['Date'].diff().dt.days
is_equally_sampled = time_intervals.nunique() == 1
fig, ax = plt.subplots(figsize=(10, 6))

if is_equally_sampled:
    # If data is equally sampled, plot as a bar chart
    ax.bar(data.index, time_intervals, width=0.5)
    ax.set_ylabel("Time Interval (in days)")
    ax.set_title("Time Intervals Between Equally Sampled Data Points")
else:
    # If data is not equally sampled, plot as a line chart
    ax.plot(time_intervals)
    ax.set_ylabel("Time Interval (in days)")
    ax.set_title("Time Intervals Between Data Points")
    ax.set_xlabel("Data Point Index")

plt.show()

#Plot for Dependent Variable VS Time

data['Date'] = pd.to_datetime(data['Date'])

print(f'\n Data types:{data.dtypes}')

# Create a plot of Open Price vs. Date
plt.figure(figsize=(15, 9))
plt.plot(data['Date'], data['Open'], label='AAPL', color='red', linewidth=1.0)
plt.title('Open Price vs. Date')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.grid(True)
plt.legend()
plt.show()

#ACF/PACF of the dependent variable

print("\n ACF Plot")
def ACF(data, lags):
    mean = np.mean(data)
    autocorrelations = []

    for lag in range(lags + 1):
        numerator = 0
        denominator = 0

        for i in range(lag, len(data)):
            numerator += (data[i] - mean) * (data[i - lag] - mean)
        for i in range(len(data)):
            denominator += (data[i] - mean) ** 2

        r = numerator / denominator
        autocorrelations.append(r)

    return autocorrelations

lags = 50
acf_values = ACF(data['Open'], lags)
left = acf_values[::-1]
right = acf_values[1:]
combine = left+right
confidence_interval= 1.96/np.sqrt(len(data))
plt.figure(figsize=(8, 5))
x_lags = list(range(-lags,lags+1))
plt.stem(x_lags, combine, markerfmt='ro', linefmt='b-', basefmt='r-')
plt.fill_between(x_lags, -confidence_interval, confidence_interval, color='lightblue', alpha=1)
plt.xlabel('Lags')
plt.ylabel('ACF Value')
plt.title(f'ACF Plot for Open price (Lags = {lags})')
plt.axhline(0, color='black')
plt.grid()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
fig = plt.figure()
plt.subplot(211)
plt.title('ACF of the open price')
sm.graphics.tsa.plot_acf(data['Open'], lags=100, ax=plt.gca())
plt.subplot(212)
plt.title('PACF of the open price')
sm.graphics.tsa.plot_pacf(data['Open'], lags=100, ax=plt.gca())
fig.tight_layout(pad=3)
plt.show()

#Heat map for correlation_matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap for Apple stock data')
heatmap.set_xlabel('Features')
heatmap.set_ylabel('Features')
plt.show()

## data Stationarity check
# ADF Test
from statsmodels.tsa.stattools import adfuller
def perform_adf_test(data, column_name):
    result = adfuller(data[column_name])
    print(f'ADF Test for {column_name}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    # Determine stationarity based on p-value
    if result[1] <= 0.05:
        print(f'Result: {column_name} is likely stationary (p-value <= 0.05)\n')
    else:
        print(f'Result: {column_name} is likely non-stationary (p-value > 0.05)\n')

# Perform ADF test for  column
perform_adf_test(data, 'Open')

#KPSS Test
from statsmodels.tsa.stattools import kpss


# Function to perform KPSS test and print the results
def perform_kpss_test(data, column_name):
    result = kpss(data[column_name])
    print(f'KPSS Test for {column_name}:')
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Lags Used: {result[2]}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'   {key}: {value}')

    # Determine stationarity based on p-value
    if result[1] >= 0.05:
        print(f'Result: {column_name} is likely stationary (p-value >= 0.05)\n')
    else:
        print(f'Result: {column_name} is likely non-stationary (p-value < 0.05)\n')

# Perform KPSS test for each column
perform_kpss_test(data, 'Open')

#Rolling Mean and Rolling Variance
def Cal_rolling_mean_var(data, column_name):
    # Initialize lists to store rolling means and variances
    rolling_means = []
    rolling_variances = []

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Loop over the number of samples in the dataset
    for i in range(1, len(data) + 1):
        # Calculate rolling mean and variance up to the current sample
        rolling_mean = data[column_name][:i].mean()
        rolling_variance = data[column_name][:i].var()

        # Append the values to the respective lists
        rolling_means.append(rolling_mean)
        rolling_variances.append(rolling_variance)

    # Plot rolling means
    axes[0].plot(rolling_means, label='Rolling Mean')
    axes[0].set_title('Rolling Mean of {}'.format(column_name))
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Mean')

    # Plot rolling variances
    axes[1].plot(rolling_variances, label='Rolling Variance', color='orange')
    axes[1].set_title('Rolling Variance of {}'.format(column_name))
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_ylabel('Variance')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

# Call the function to calculate and plot rolling mean and variance for "Sales" column
Cal_rolling_mean_var(data, 'Open')

#Differencing
print("=======First Order Differencing======")
differenced_data = copy.deepcopy(data)
for i in range(1, len(differenced_data)):
    differenced_data.at[i, 'Open_1st_Difference'] = differenced_data.at[i, 'Open'] - differenced_data.at[i - 1, 'Open']

print("\n  After First order differencing\n", differenced_data)
perform_adf_test(differenced_data.dropna(), 'Open_1st_Difference')
perform_kpss_test(differenced_data.dropna(),'Open_1st_Difference')
Cal_rolling_mean_var(differenced_data, 'Open_1st_Difference')

print("=======Second Order Differencing======")
for i in range(1, len(differenced_data)):
    differenced_data.at[i, 'Open_2nd_Difference'] = differenced_data.at[i, 'Open_1st_Difference'] - differenced_data.at[i - 1, 'Open_1st_Difference']

print("\n  After Second order differencing\n", differenced_data)
perform_adf_test(differenced_data.dropna(), 'Open_2nd_Difference')
perform_kpss_test(differenced_data.dropna(), 'Open_2nd_Difference')
Cal_rolling_mean_var(differenced_data, 'Open_2nd_Difference')

data['Date'] = pd.to_datetime(data['Date'])

# Create a plot of Open Price vs. Date
plt.figure(figsize=(15, 9))
plt.plot(differenced_data['Date'], differenced_data['Open_2nd_Difference'], label='AAPL', color='red', linewidth=1.0)
plt.title('Open Price vs. Date')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.grid(True)
plt.legend()
plt.show()


#STL Decomposition
from statsmodels.tsa.seasonal import STL, seasonal_decompose
Temp = data['Open']
Temp = pd.Series(np.array(Temp), index=pd.date_range('1980-12-12',
                                                     periods=len(Temp),
                                                     freq = 'D'),)
STL = STL(Temp)
res = STL.fit()
plt.figure(figsize=(8, 6))

# Plot the original time series
plt.subplot(411)
plt.plot(res.observed, color='blue')
plt.title('Original Time Series')

# Plot the trend component
plt.subplot(412)
plt.plot(res.trend, color='green')
plt.title('Trend Component')

# Plot the seasonal component
plt.subplot(413)
plt.plot(res.seasonal, color='red')
plt.title('Seasonal Component')

# Plot the residual component
plt.subplot(414)
plt.plot(res.resid, color='purple')
plt.title('Residual Component')

plt.tight_layout()
plt.show()

#Dtrend and Seasonally Adjusted
T = res.trend
S = res.seasonal
R = res.resid

seasonal_adjusted = Temp-S
Detrended = Temp - T
plt.figure(figsize=(8,6))
plt.plot(Temp, label='Actual Data')
plt.plot(seasonal_adjusted, label='Adjusted Seasonal Data')
plt.title("Actual Data vs Adjusted Seasonal Data")
plt.xlabel("Date")
plt.ylabel('open price')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(Temp, label='Actual Data')
plt.plot(Detrended, label='Detrended Data')
plt.title("Actual Data vs Detrended Data")
plt.xlabel("Date")
plt.ylabel('Open price')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()


def str_trend(T, S, R):
    FT = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(T + R)))
    print(f'\n Strength of trend for the raw data is {100 * FT:.3f}%')


str_trend(T, S, R)

def str_seasonal(T, S, R):
    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'Strength of seasonality for the raw data is {100 * FS:.3f}%')


str_seasonal(T, S, R)

#Split the dataset into train set (80%) and test set (20%).
from sklearn.model_selection import train_test_split

# Features (X) and Target Variable (y)
#X = differenced_data.drop(columns=['Open', 'Date'])  # Features excluding 'Open' and 'Date' columns
X = differenced_data[['High','Low','Close','Adj Close','Volume']]
X1 = differenced_data[['High','Low','Close','Adj Close','Volume','Open']]
y = differenced_data['Open_2nd_Difference']  # Target variable 'Open'
X = X.iloc[2:,:]
y = y[2:]



# Splitting the dataset into 80% train set and 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle = False)
date_train,date_test = train_test_split(differenced_data['Date'][2:], test_size=0.2, random_state=42,shuffle = False)
yt,yf = train_test_split(data['Open'],shuffle=False,test_size=0.2)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"date_train_test shape: {date_train.shape}, y_test shape: {date_test.shape}")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Holt-Winters method fitting on the train dataset
holtt = ExponentialSmoothing(yt, trend='mul', damped_trend=True, seasonal='mul', seasonal_periods=12).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

# Plotting the results
plt.plot(date_train, yt[2:], label='Train Data')
plt.plot(date_test, yf, label='Test Data')
plt.plot(date_test, holtf, label='Holt Winter Forecast')
plt.title('Holt Winter (Prediction Plot)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#============feature Selection====================
def lse(Y, Z):
    return np.linalg.inv(Y.T.dot(Y)).dot(Y.T).dot(Z)
from numpy import linalg as LA
print("Feature Selection...\n")
H = np.matmul(X1.T, X1)
print(f'Shape of H is {H.shape}')
s, d, v = np.linalg.svd(H)
print(f'Singular Values {d}')
print(f'The condition number is {LA.cond(X)}')
print('unknown coefficients :',lse(X_train, y_train))

print("===================================OLS MODEL=================================")

# X = sm.add_constant(X_train)
model_1 = sm.OLS(y_train, X_train)
output1 = model_1.fit()

print(output1.summary())

print("=========================== Volume dropped ==================================")

X_train2 = X_train.drop(['Volume'],axis=1)
model2 = sm.OLS(y_train,X_train2)
output2 = model2.fit()
print(output2.summary())

print("=========================== Adj Close dropped ==================================")

X_train3 = X_train2.drop(['Adj Close'],axis=1)
model3 = sm.OLS(y_train,X_train3).fit()
# output3 = model3.fit()
print(model3.summary())

X_test_new = X_test.drop(['Volume','Adj Close'],axis=1)
y_pred_ols = model3.predict(X_test_new)

def inverse_difference(original_data, forecast, interval=1):
    new_data = np.zeros(len(original_data))
    for i in range(1, len(forecast)):
        new_data[i] = forecast[i - interval] + original_data[i - interval]
    new_data = new_data[1:]
    return new_data
open = data['Open']
y_pred_ols_inv = inverse_difference(open[len(y_train)+1:].values,np.array(y_pred_ols),1)

# print(len(date_test))
# print(len(y_pred_ols_inv))
plt.plot(date_test, open[len(y_train)+2:].values.flatten(), label='Test Data')
plt.plot(date_test, y_pred_ols_inv, label='OLS Method Forecast')
plt.title('OLS (Prediction Plot) --> open vs time')
plt.xlabel('date')
plt.ylabel('open')
plt.legend()
plt.show()
print("\n")

plt.plot(date_test, y_test.values.flatten(), label='Test Data')
plt.plot(date_test, y_pred_ols.values.flatten(), label='OLS Method Forecast')
plt.title('OLS (Prediction Plot) --> open vs Time ')
plt.xlabel('date')
plt.ylabel('open price')
plt.legend()
plt.show()
print("\n")

ols_err = y_test-y_pred_ols
# Diagnostic Testing
print('Mean of residual error (OLS) method :', np.mean(ols_err))
print(f'variance of residual error (OLS) method :',np.var(ols_err))

#Basic Methods
#Average method
Average_train_df = pd.DataFrame()
Average_test_df = pd.DataFrame()
Average_train_df['open'] = copy.deepcopy(y_train)
Average_test_df['open'] = copy.deepcopy(y_test)

x = y_train.values
y = y_test.values

x_pred_1_step = [np.nan] * len(x)
y_pred_h_step = [np.nan] * len(y)
errors_1_step = [np.nan] * len(x)
errors_h_step = []

# perform one-step ahead prediction and calculate errors
for t in range(2, len(x)):
    x_pred_1_step[t] = np.mean(x[:t])
    errors_1_step[t] = (x[t] - x_pred_1_step[t])

# Perform h-step forecast and calculate errors

for h in range(len(y)):
    y_pred_h_step[h] = np.mean(x[:len(x)])
    errors_h_step.append(y[h] - y_pred_h_step[h])

Average_train_df['1_step_prediction'] = x_pred_1_step
Average_train_df['e'] = errors_1_step
Average_train_df['squared_error'] = Average_train_df['e'] ** 2
print("\nAverage calculation for train set")
print(Average_train_df)

Average_test_df['h_step_prediction'] = y_pred_h_step
Average_test_df['h_error'] = errors_h_step
Average_test_df['squared_error'] = Average_test_df['h_error'] ** 2
print("\n Average calculation for test set")
print(Average_test_df)

plt.plot(date_train, y_train, label='Train Data')
plt.plot(date_test, y_test, label='Test Data')
plt.plot(date_test,Average_test_df['h_step_prediction'], label='Average Method Forecast')
plt.title('Average Method (Prediction Plot) --> Open Price vs time')
plt.xlabel('Time')
plt.ylabel('open')
plt.legend()
plt.show()

# Naive method
print("/n")
Naive_train_df = pd.DataFrame()
Naive_test_df = pd.DataFrame()
Naive_train_df['open'] = copy.deepcopy(y_train)
Naive_test_df['open'] = copy.deepcopy(y_test)


x = y_train.values
y = y_test.values

x_pred_1_step = [np.nan] * len(x)
y_pred_h_step = [np.nan] * len(y)
errors_1_step = [np.nan] * len(x)
errors_h_step = []

# perform one-step ahead prediction and calculate errors
for t in range(2, len(x)):
    x_pred_1_step[t] = x[t - 1]
    errors_1_step[t] = (x[t] - x_pred_1_step[t])

# Perform h-step forecast and calculate errors

for h in range(len(y)):
    y_pred_h_step[h] = x[t]
    errors_h_step.append(y[h] - y_pred_h_step[h])

Naive_train_df['1_step_prediction'] = x_pred_1_step
Naive_train_df['e'] = errors_1_step
Naive_train_df['squared_error'] = Naive_train_df['e'] ** 2
print("\nNaive calculation for train set")
print(Naive_train_df)
print("\n")

Naive_test_df['h_step_prediction'] = y_pred_h_step
Naive_test_df['e'] = errors_h_step
Naive_test_df['squared_error'] = Naive_test_df['e'] ** 2
print("Naive calculation for test set")
print(Naive_test_df)

plt.plot(date_train, y_train, label='Train Data')
plt.plot(date_test, y_test, label='Test Data')
plt.plot(date_test,Naive_test_df['h_step_prediction'], label='Naive Method Forecast')
plt.title('Naive Method (Prediction Plot) --> Open price vs time')
plt.xlabel('Time')
plt.ylabel('open')
plt.legend()
plt.show()


#Drift Method


print("/n")
Drift_train_df = pd.DataFrame()
Drift_test_df = pd.DataFrame()
Drift_train_df['open'] = copy.deepcopy(y_train)
Drift_test_df['open'] = copy.deepcopy(y_test)


x = y_train.values
y = y_test.values

x_pred_1_step = [np.nan] * len(x)
y_pred_h_step = [np.nan] * len(y)
errors_1_step = [np.nan] * len(x)
errors_h_step = []
h=1
# perform one-step ahead prediction and calculate errors
for t in range(2, len(x)):
    num = x[t-1]-x[0]
    den = t-1
    x_pred_1_step[t] = x[t-1] + (h * num / den)
    errors_1_step[t] = (x[t] - x_pred_1_step[t])

# Perform h-step forecast and calculate errors
for h in range(len(y)):
    T = len(x)
    h_num = x[T-1]-x[0]
    h_den = T-1
    H = h+1
    y_pred_h_step[h] = x[t]+(H *h_num/h_den)
    errors_h_step.append(y[h] - y_pred_h_step[h])

Drift_train_df['1_step_prediction'] =x_pred_1_step
Drift_train_df['e'] = errors_1_step
Drift_train_df['squared_error'] = Drift_train_df['e'] ** 2
print("\nDrift calculation for train set")
print(Drift_train_df)

Drift_test_df['h_step_prediction'] =y_pred_h_step
Drift_test_df['e'] = errors_h_step
Drift_test_df['squared_error'] = Drift_test_df['e'] ** 2
print("\nDrift calculation for train set")
print(Drift_test_df)

plt.plot(date_train, y_train, label='Train Data')
plt.plot(date_test, y_test, label='Test Data')
plt.plot(date_test,Drift_test_df['h_step_prediction'], label='Drift Method Forecast')
plt.title('Drift Method (Prediction Plot) --> Open price vs time')
plt.xlabel('Time')
plt.ylabel('open')
plt.legend()
plt.show()

#SES Method
SES_train_df = pd.DataFrame()
SES_test_df = pd.DataFrame()
SES_train_df['open'] = copy.deepcopy(y_train)
SES_test_df['open'] = copy.deepcopy(y_test)


x = y_train.values
y = y_test.values

x_pred_1_step = [np.nan] * len(x)
y_pred_h_step = [np.nan] * len(y)
errors_1_step = [np.nan] * len(x)
errors_h_step = []

alpha = 0.5
IC = x[0]
# perform one-step ahead prediction and calculate errors
for t in range(1, len(x)):
    x_pred_1_step[t] = x[t - 1] * alpha + (1 - alpha) * IC
    IC = x_pred_1_step[t]
    errors_1_step[t] = (x[t] - x_pred_1_step[t])

# Perform h-step forecast and calculate errors

for h in range(len(y)):
    T = len(x)
    y_pred_h_step[h] = x[T - 1] * alpha + (1 - alpha) * IC
    errors_h_step.append(y[h] - y_pred_h_step[h])

SES_train_df['1_step_prediction'] = x_pred_1_step
SES_train_df['e'] = errors_1_step
SES_train_df['squared_error'] = SES_train_df['e'] ** 2
print("\nSES calculation for train set")
print(SES_train_df)


SES_test_df['h_step_prediction'] = y_pred_h_step
SES_test_df['e'] = errors_h_step
SES_test_df['squared_error'] = SES_test_df['e'] ** 2
print("SES calculation for train set")
print(SES_test_df)

plt.plot(date_train, y_train, label='Train Data')
plt.plot(date_test, y_test, label='Test Data')
plt.plot(date_test,SES_test_df['h_step_prediction'], label='SES Method Forecast')
plt.title('SES Method (Prediction Plot) --> Open price vs time')
plt.xlabel('Time')
plt.ylabel('open')
plt.legend()
plt.show()



#ARMA Model
print("\nARMA Model...\n")
def calculate_gpac_values(ry, nb, na):
    na += 1  # Adjusting for zero-indexing
    gpac_values = np.empty((nb, na))  # Initialize the GPAC values array

    for ar_order in range(1, na):
        numerator_matrix = np.empty((ar_order, ar_order))
        denominator_matrix = np.empty((ar_order, ar_order))
        for ma_order in range(nb):
            for i in range(ar_order):
                for j in range(ar_order):
                    if j < ar_order - 1:
                        numerator_matrix[i][j] = ry[abs(ma_order + (i - j))]
                        denominator_matrix[i][j] = ry[abs(ma_order + (i - j))]
                    else:
                        numerator_matrix[i][j] = ry[abs(ma_order + i + 1)]
                        denominator_matrix[i][j] = ry[abs(ma_order + (i - j))]

            numerator_det = round(np.linalg.det(numerator_matrix), 6)
            denominator_det = round(np.linalg.det(denominator_matrix), 6)

            if denominator_det == 0.0:
                gpac_values[ma_order][ar_order] = np.inf
            else:
                gpac_values[ma_order][ar_order] = round((numerator_det / denominator_det), 3)

    gpac_dataframe = pd.DataFrame(gpac_values[:, 1:])  # Exclude the first column
    gpac_dataframe.columns = [f'{i}' for i in range(1, na)]

    return gpac_dataframe

def plot_gpac_heatmap(gpac_dataframe, process):
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_dataframe, annot=True)
    plt.title(f'GPAC Table {process}')
    plt.xlabel("AR order (na)")
    plt.ylabel("MA order (nb)")
    plt.show()


# from statsmodels.tsa.stattools import acf
# acf_values = acf(differenced_data['Open_2nd_Difference'].dropna().values, lags)
# acf_left = acf_values[::-1]
# acf_right = acf_values[1:]
# combine_gpac = acf_left + acf_right


from statsmodels.tsa.stattools import acf
y2 = differenced_data['Open_2nd_Difference']

acf_values = acf(y2[2:], 100)


gpac =calculate_gpac_values(acf_values,7,7)
plot_gpac_heatmap(gpac, 'Open price')



import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
fig = plt.figure()
plt.subplot(211)
plt.title('ACF of the generated data')
sm.graphics.tsa.plot_acf(differenced_data['Open_2nd_Difference'].dropna(), lags=100, ax=plt.gca())
plt.subplot(212)
plt.title('PACF of the generated data')
sm.graphics.tsa.plot_pacf(differenced_data['Open_2nd_Difference'].dropna(), lags=100, ax=plt.gca())
fig.tight_layout(pad=3)
plt.show()

from pmdarima import auto_arima
stepwise_fit = auto_arima(differenced_data['Open_2nd_Difference'].dropna(), trace=True,
suppress_warnings=True)


# from sktime.forecasting.arima import AutoARIMA
# y = differenced_data['Open_2nd_Difference'].dropna()
# # p--->AR non-seasonal
# # q---->MA non-seasonal
# # P --->AR seasonal
# # W---->MA seasonal
# # d --- order of non-seasonal differencing
# # D --- order of seasonal differencing
# forecaster =  AutoARIMA(start_p = 0,
#                         max_p = 20,
#                         start_q = 0,
#                         max_q = 20,
#                         max_d = 5,
#
#                         stationary = True,
#                         n_fits = 20,
#                         stepwise = False
# )
# forecaster = forecaster.fit(y)
# print(forecaster.summary())


# ARMA MODEL
print("ARMA model with na = 5 and nb = 0")
na = 5
nb = 0

Arma_model = sm.tsa.ARIMA(y_train, order=(na, 0, nb), trend='c').fit()

print(Arma_model.summary())

# Print AR and MA coefficients
for i in range(na):
    print(f'The AR Coefficient a{i+1} is: {Arma_model.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient b{i+1} is: {Arma_model.params[i+na]}')

# Forecast into the future
prediction = Arma_model.predict(start=0, end=8373)  # Adjust the range as needed

# Plot actual vs. predicted
fig = plt.figure(figsize=(12, 6))
plt.title('ARMA Predictions')
plt.plot(y_train, label='Actual', color='blue')
plt.plot(prediction, label='Predicted', color='red')
plt.title('Open Price vs time(ARMA(5,0)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Open Price')
plt.legend()
plt.show()


Arma_df = pd.DataFrame()
Arma_df['y_train'] = y_train
Arma_df['y_train+1'] = prediction
Arma_df['error'] = Arma_df['y_train'] - Arma_df['y_train+1']
Arma_df['error_square'] = Arma_df['error']**2

Arma_df.reset_index(drop=True, inplace=True)
print(Arma_df)

lags = 100
acf_values = ACF(Arma_df['error'], lags)
left = acf_values[::-1]
right = acf_values[1:]
combine = left+right
confidence_interval= 1.96/np.sqrt(len(Arma_df))
plt.figure(figsize=(8, 5))
x_lags = list(range(-lags,lags+1))
plt.stem(x_lags, combine, markerfmt='ro', linefmt='b-', basefmt='r-')
plt.fill_between(x_lags, -confidence_interval, confidence_interval, color='lightblue', alpha=1)
plt.xlabel('Lags')
plt.ylabel('ACF Value')
plt.title(f'ACF Plot for arma data errors (Lags = {lags})')
plt.axhline(0, color='black')
plt.grid()
plt.show()


plt.plot(date_train[:2500],Arma_df['y_train'][:2500], label = 'train set')
plt.plot(date_train[:2500],Arma_df['y_train+1'][:2500], label = '1-step prediction')
plt.title('Open Price vs time(ARMA(5,0)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Open Price')
plt.legend()
plt.tight_layout()
plt.show()


# diagnostic testing
from scipy.stats import chi2

print('parameters for confidence intervals :',Arma_model.conf_int())

poles = []
for i in range(na):
    poles.append(-(Arma_model.params[i]))

print('zero/cancellation:')
zeros = []
for i in range(nb):
    zeros.append(-(Arma_model.params[i+na]))

print(f'zeros : {zeros}')
print(f'poles : {poles}')
Q = len(y_train)*np.sum(np.square(acf_values[lags:]))
Degree_of_freedom = lags-na-nb
alpha = 0.01
chi_critical = chi2.ppf(1-alpha, Degree_of_freedom)
print('Chi Squared test results')
if Q<chi_critical:
    print(f'The residuals is white,  and the chi_square values is :{Q}')
else:
    print(f'The residual is NOT white, chi squared value :{Q}')


# =========== h step prediction ================
forecast = Arma_model.forecast(steps=len(y_test))

Arma_df_forecast = pd.DataFrame()
Arma_df_forecast['y_test'] = y_test
Arma_df_forecast['y_test+h_step'] = forecast
Arma_df_forecast['forecast_error'] = Arma_df_forecast['y_test'] - Arma_df_forecast['y_test+h_step']
Arma_df_forecast['forecast_error_square'] = Arma_df_forecast['forecast_error']**2

plt.plot(date_test,y_test, label = 'test set')
plt.plot(date_test,forecast, label = 'h-step prediction')
plt.title('Open Price vs time(ARMA(5,0)) - Forecasting  plot')
plt.xlabel('time')
plt.ylabel('Open Price')
plt.legend()
plt.tight_layout()
plt.show()

print(Arma_df_forecast)
res_var = Arma_df['error'].var()
print(f'variance of residual error : {res_var}')
forecast_var = Arma_df_forecast['forecast_error'].var()
print(f'variance of forecast error : {forecast_var}')

#
# #=====Arma(4,5)
# Arma_model_2 = sm.tsa.ARIMA(y_train, order=(4, 0, 5), trend='c').fit()
#
# print(Arma_model_2.summary())
#
# # Print AR and MA coefficients
# for i in range(na):
#     print(f'The AR Coefficient a{i+1} is: {Arma_model_2.params[i]}')
#
# for i in range(nb):
#     print(f'The MA Coefficient b{i+1} is: {Arma_model_2.params[i+na]}')
#
# # Forecast into the future
# prediction_2 = Arma_model_2.predict(start=0, end=8373)  # Adjust the range as needed
#
# # Plot actual vs. predicted
# fig = plt.figure(figsize=(12, 6))
# plt.title('ARMA Predictions')
# plt.plot(y_train, label='Actual', color='blue')
# plt.plot(prediction_2, label='Predicted', color='red')
# plt.legend()
# plt.show()
#
#
# Arma_df_2 = pd.DataFrame()
# Arma_df_2['y_train'] = y_train
# Arma_df_2['y_train+1'] = prediction_2
# Arma_df_2['error'] = Arma_df_2['y_train'] - Arma_df_2['y_train+1']
# Arma_df_2['error_square'] = Arma_df_2['error']**2
#
# Arma_df_2.reset_index(drop=True, inplace=True)
# print(Arma_df)
#
# lags = 100
# acf_res = acf(Arma_df_2['error'].values, nlags= lags)
# plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf_res[::-1])[:-1],acf_res)), linefmt='grey', markerfmt='o')
# m = 1.96 / np.sqrt(100)
# plt.axhspan(-m, m, alpha=.2, color='blue')
# plt.title("ACF Plot of residual error (ARMA(4,5))")
# plt.xlabel("Lags")
# plt.ylabel("ACF values")
# plt.grid()
# plt.legend(["ACF"], loc='upper right')
# plt.show()
#
#
#
#
# # diagnostic testing
# from scipy.stats import chi2
#
# print('parameters for confidence intervals :',Arma_model_2.conf_int())
#
# poles = []
# for i in range(na):
#     poles.append(-(Arma_model_2.params[i]))
#
# print('zero/cancellation:')
# zeros = []
# for i in range(nb):
#     zeros.append(-(Arma_model_2.params[i+na]))
#
# print(f'zeros : {zeros}')
# print(f'poles : {poles}')
# Q = len(y_train)*np.sum(np.square(acf_res[lags:]))
# Degree_of_freedom = lags-4-5
# alpha = 0.01
# chi_critical = chi2.ppf(1-alpha, Degree_of_freedom)
# print('Chi Squared test results')
# if Q<chi_critical:
#     print(f'The residuals is white,  and the chi_square values is :{Q}')
# else:
#     print(f'The residual is NOT white, chi squared value :{Q}')

#==============ARIMA===========================
na = 5
d = 2
nb = 0

Arima_model = sm.tsa.ARIMA(y_train, order=(na, d, nb)).fit()

print(Arima_model.summary())

# Print AR and MA coefficients
for i in range(na):
    print(f'The AR Coefficient a{i+1} is: {Arima_model.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient b{i+1} is: {Arima_model.params[i+na]}')

# Forecast into the future
Arima_prediction = Arima_model.predict(start=0, end=8373)  # Adjust the range as needed

# Plot actual vs. predicted
fig = plt.figure(figsize=(12, 6))
plt.title('ARIMA Predictions')
plt.plot(y_train, label='Actual', color='blue')
plt.plot(Arima_prediction, label='Predicted', color='red')
plt.title('Open Price vs time(ARIMA(5,2,0)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Open Price')
plt.legend()
plt.show()


Arima_df = pd.DataFrame()
Arima_df['y_train'] = y_train
Arima_df['y_train+1'] = Arima_prediction
Arima_df['error'] = Arima_df['y_train'] - Arima_df['y_train+1']
Arima_df['error_square'] = Arima_df['error']**2


Arima_df.reset_index(drop=True, inplace=True)
print(Arima_df)

lags = 100
acf_values = ACF(Arima_df['error'], lags)
left = acf_values[::-1]
right = acf_values[1:]
combine = left+right
confidence_interval= 1.96/np.sqrt(len(Arima_df))
plt.figure(figsize=(8, 5))
x_lags = list(range(-lags,lags+1))
plt.stem(x_lags, combine, markerfmt='ro', linefmt='b-', basefmt='r-')
plt.fill_between(x_lags, -confidence_interval, confidence_interval, color='lightblue', alpha=1)
plt.xlabel('Lags')
plt.ylabel('ACF Value')
plt.title(f'ACF Plot for arima data errors (Lags = {lags})')
plt.axhline(0, color='black')
plt.grid()
plt.show()


plt.plot(date_train[:2500],Arima_df['y_train'][:2500], label = 'train set')
plt.plot(date_train[:2500],Arima_df['y_train+1'][:2500], label = '1-step prediction')
plt.title('Open Price vs time(ARIMA(5,2,0)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Open Price')
plt.legend()
plt.tight_layout()
plt.show()


# diagnostic testing
from scipy.stats import chi2

print('parameters for confidence intervals :',Arima_model.conf_int())

poles = []
for i in range(na):
    poles.append(-(Arima_model.params[i]))

print('zero/cancellation:')
zeros = []
for i in range(nb):
    zeros.append(-(Arima_model.params[i+na]))

print(f'zeros : {zeros}')
print(f'poles : {poles}')
Q = len(y_train)*np.sum(np.square(acf_values[lags:]))
Degree_of_freedom = lags-na-nb
alpha = 0.01
chi_critical = chi2.ppf(1-alpha, Degree_of_freedom)
print('Chi Squared test results')
if Q<chi_critical:
    print(f'The residuals is white,  and the chi_square values is :{Q}')
else:
    print(f'The residual is NOT white, chi squared value :{Q}')


# =========== h step prediction ================
Arima_forecast = Arima_model.forecast(steps=len(y_test))

Arima_df_forecast = pd.DataFrame()
Arima_df_forecast['y_test'] = y_test
Arima_df_forecast['y_test+h_step'] = Arima_forecast
Arima_df_forecast['forecast_error'] = Arima_df_forecast['y_test'] - Arima_df_forecast['y_test+h_step']
Arima_df_forecast['forecast_error_square'] = Arima_df_forecast['forecast_error']**2

plt.plot(date_test,Arima_df_forecast['y_test'], label = 'test set')
plt.plot(date_test,Arima_df_forecast['y_test+h_step'], label = 'h-step prediction')
plt.title('Open Price vs time(ARIMA(5,2,0)) - Forecasting  plot')
plt.xlabel('time')
plt.ylabel('Open Price')
plt.legend()
plt.tight_layout()
plt.show()

print(Arima_df_forecast)
# res_var = Arima_df_forecast['error'].var()
print(f"Variance of residual error: {Arima_df['error'].var()}")
print(f"Variance of forecast error: {Arima_df_forecast['forecast_error'].var()}")

