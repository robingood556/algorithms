import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from texttable import Texttable
from scipy import stats
from math import inf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics as m
from sklearn import linear_model
from sklearn.linear_model import  LassoLarsIC

import scipy as sp

from prettytable import PrettyTable
from scipy.optimize import curve_fit, minimize


full_df = pd.read_csv('heart.csv')
print(full_df.head(10))

df = full_df[ [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak'
          ] ]


print(df.head(10))

cols = df.columns.to_list()[1:-1]
for col in cols:
    sns.jointplot(data=df, x=col, y='oldpeak', hue='age', kind='kde', dropna=True)

#plt.show()

t = Texttable()

title = [ ['Column name', 'mathematical expectation', 'variance'] ]
rows = [ [col, df.loc[:, col].mean(), df.loc[:, col].var()] for col in df.columns[1:] ]
title.extend(rows)

t.set_deco(Texttable.HEADER)
t.set_cols_align(["l", "r", "r"])
t.add_rows(title)

print(t.draw())

plt.figure(figsize=(15, 20))
plt.suptitle("Non-parametric estimation of conditional distributions", fontsize=20, y=1.01)

num_columns = len(df.columns[1:])
for n, col in enumerate(df.columns[1:]):
    ax = plt.subplot(num_columns // 2 if num_columns % 2 == 0 else num_columns // 2 + 1, 2, n + 1)

    sns.kdeplot(df.loc[df['oldpeak'] < 1.0 , col],
                shade=True,
                label='0',
                warn_singular=False,
                ax=ax)
    sns.kdeplot(df.loc[df['oldpeak'] > 1.0 , col],
                shade=True,
                label="1",
                warn_singular=False,
                ax=ax)

    ax.legend(title='oldpeak', loc='upper right')
    ax.set_title(col.upper(), fontsize=12, color='black')
    ax.set(xlabel=None)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
plt.show()

stat_df = {
    'Feature': [],
    'Corr coefficient': [],
    'Significance level': [],
    'Confidence interval': []
}

clean_df = df.dropna()
for col in df.columns[1:]:
    # Confidence interval for the correlation coefficient
    r, p = stats.pearsonr(clean_df[col].apply(lambda x: float(x)),
                          clean_df.oldpeak.apply(lambda x: 0 if x > 1.0 else 1))
    r_z = np.arctanh(r)  # matches Fisher transform

    # Corresponding standard deviation
    se = 1 / np.sqrt(clean_df[col].size - 3)
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se

    stat_df['Feature'].append(col)
    stat_df['Corr coefficient'].append(round(r, 5))
    stat_df['Significance level'].append(0 if p > 0.05 else 1)
    stat_df['Confidence interval'].append(f'[{lo_z} ... {hi_z}]')
stat_df = pd.DataFrame(stat_df)

print(stat_df.to_string())


df.oldpeak= df.oldpeak.apply(lambda x: 0 if x > 1.0 else 1)
plt.figure(figsize=(15, 10))
plt.title('Multivariate correlation matrix', fontsize=20, y=1.01)
sns.heatmap(df.corr(), annot=True, fmt= '.1f', cmap = 'Reds')

plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 1:], diagonal="kde", figsize=(20,20))
plt.tight_layout()
plt.show()

ax = df[['trestbps', 'chol', 'thalach']].plot(figsize=(20,8))
ax.legend(loc='upper right')

plt.show()

_ = df.hist(sharex=False, sharey=False, grid=False, figsize=(20,10))
plt.tight_layout()
plt.show()

X = full_df.drop(['trestbps'], axis=1)
y = full_df['trestbps']
scaler = StandardScaler()
scaler.fit(X)
standardisedX = scaler.transform(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(standardisedX, y, test_size=0.2, random_state=42)

reg = LinearRegression()
# Train a linear regression model
reg.fit(X_train, y_train)
# Forecast on a test sample
y_pred = reg.predict(X_test)
params = np.append(reg.intercept_,reg.coef_)
# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

row1 = [ ['Least Squares model', '-', str(mse), str(mae), str(r2), str(reg.coef_)] ]

print('Mean absolute error = ', mae)
print('Mean squared error = ', mse)
print('R2 score = ', r2)

l_A = -1
l_MSE = inf
l_MAE = inf
l_var = -1
l_coef = (0, 0, 0, 0, 0, 0)

for alpha in np.arange(0.0,1.001,0.001):
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    l_coef = clf.coef_
    if mse < l_MSE and r2 > l_var:
        l_A = alpha
        l_MSE = mse
        l_MAE = mae
        l_var = r2
        l_coef = l_coef

row2 = [ ['Best Lasso model', str(l_A), str(l_MSE), str(l_MAE), str(l_var), str(l_coef)] ]

print('Mean absolute error = ', l_MAE)
print('Mean squared error = ', l_MSE)
print('R2 score = ', l_var)

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X_train, y_train)

# Forecast on a test sample
y_pred_lasso_aic = model_aic.predict(X_test)
params = np.append(model_aic.intercept_, model_aic.coef_)
# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred_lasso_aic)
mse = mean_squared_error(y_test, y_pred_lasso_aic)
r2 = r2_score(y_test, y_pred_lasso_aic)

row3 = [ ['LassoLarsIC', '-', str(mse), str(mae), str(r2), str(model_aic.coef_)] ]

print('Mean absolute error with aic lasso = ', mae)
print('Mean squared error with aic lasso = ', mse)
print('R2 score with aic lasso = ', r2)

t = Texttable()
title = [ ['Type', 'Alpha', 'MSE', 'MAE', 'VAR'] ]

for row in [row1, row2, row3]:
    title.extend([row[0][:-1]])
t.set_cols_align(['l', 'c', 'c', 'c', 'c'])
t.add_rows(title)

print(t.draw())

x = []
for i in range(len(y_test)):
    x.append(i)

#graph of real and predicted values
plt.scatter(x, y_test, label = u'The real temperature')
plt.scatter(x, y_pred, label = u'Predicted by the linear model')
plt.title(u'Real values of temperature and predicted by the linear model')
plt.legend(loc="center right",borderaxespad=0.1, bbox_to_anchor=(1.7, 0.5))
plt.xlabel(u'id')
plt.ylabel(u'Temp')

plt.show()



X = full_df.drop(['trestbps'], axis=1)
y = full_df['trestbps']
scaler = StandardScaler()
scaler.fit(X)
standardisedX = scaler.transform(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(standardisedX, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

residuals = full_df['trestbps'] - model.predict(standardisedX)
residuals.describe()
table = PrettyTable()
table.field_names = ["SL", "CV", 'H0']

result = sp.stats.anderson(residuals, dist='norm')
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        table.add_row([sl, cv, 'data looks normal (fail to reject H0)'])
    else:
        table.add_row([sl, cv, "data doesn't look normal (fail to reject H0)"])

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
sns.kdeplot(residuals, shade=True, color='red', ax=axes[0])
sp.stats.probplot(residuals, dist="norm", plot=axes[1])
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

print('Statistic: %.3f' % result.statistic)
print(table)

ks = sp.stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.var()))
print(ks)
if ks[1]==0:
    print('Residuals are not distributed normally')



