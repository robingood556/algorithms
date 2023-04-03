import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp
import warnings
warnings.simplefilter("ignore", UserWarning)

from datetime import datetime
from prettytable import PrettyTable
from scipy.optimize import curve_fit, minimize


df = pd.read_csv('heart.csv', engine='python')

df = df[[
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
    ]]

print(df)

col_names = list(df.columns)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
col_idx = 1
for i in range(2):
    for j in range(2):
        axes[i][j].scatter(df.index, df[col_names[col_idx]], c='red', linewidths=0.01)
        axes[i][j].set_xlabel(col_names[col_idx], fontsize = 15)
        col_idx += 1


plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
col_idx = 1
for i in range(2):
    for j in range(2):
        col_name = col_names[col_idx]
        df=df.dropna()
        kernel = sp.stats.gaussian_kde(df[col_name])
        min_amount, max_amount = df[col_name].min(), df[col_name].max()
        x = np.linspace(min_amount, max_amount, len(df[col_name]))
        kde_values = kernel(x)
        sns.histplot(df[col_name], kde=False, bins=30, stat='density', ax=axes[i, j], palette='Set3', color = 'blue')
        axes[i, j].plot(x, kde_values, c = 'red')
        col_idx += 1

plt.show()

col_names = [
    'age',
    'trestbps',
    'chol',
    'thalach'
]

fig, axes = plt.subplots(len(col_names), figsize=(16, 16))

table = PrettyTable()
table.field_names = ["column name", "m.expectation", "median", "variance", "s.deviation"]

for col_idx in range(len(col_names)):
    # Calculation of sample mean, variance, standard deviation, median
    df = df.dropna()
    col_name = col_names[col_idx]
    mean = df[col_name].mean()
    var = df[col_name].var()
    std = df[col_name].std()
    median = df[col_name].median()

    table.add_row([col_names[col_idx], mean, median, var, std])

    whis_p = 1.5

    boxplot = df.boxplot(column=col_name, whis=whis_p, vert=False, ax=axes[col_idx], color='blue')

print(table)

plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
col_idx = 0

mle_params_list = []
lse_params_list = []

table = PrettyTable()
table.field_names = ["column name", "MLE", "LSE"]

for i in range(2):
    for j in range(2):
        # Determination of the parameters of the distribution
        col_name = col_names[col_idx]
        # Prepare data for least squares method, but do not show this histogramm
        hist_data = axes[i, j].hist(df[col_name], density=True, bins=50, color='blue', visible=False)
        sns.histplot(df[col_name], kde=False, bins=30, stat='density', ax=axes[i, j], palette='Set3', color='blue')
        min_amount, max_amount = df[col_name].min(), df[col_name].max()

        #  The distribution parameters are determined using the fit function based on least squares method
        # Prepare data for least squares
        hist_bins = hist_data[1][:-1]  # remove last element
        delta = hist_bins[1] - hist_bins[0]
        hist_bins += delta / 2  # Take centres of bins
        hist_vals = hist_data[0]  # Get the height of bins for least squares

        x = np.linspace(min_amount, max_amount)

        # Do MLE
        if col_name == 'age' or col_name == 'trestbps':
            # The lognorm distribution parameters are determined using the fit function based on the maximum likelihood method
            mle_params = sp.stats.lognorm.fit(df[col_name], loc=1.1, scale=1.1)  # Make initial Guess
            pdf_mle = sp.stats.lognorm.pdf(x, *mle_params)

        elif col_name == 'chol' or "thalach":
            mle_params = sp.stats.chi2.fit(df[col_name], loc=1.1, scale=1.1)
            pdf_mle = sp.stats.chi2.pdf(x, *mle_params)

        axes[i, j].plot(x, pdf_mle, color='r', label="MLE")

        # Do LSE
        if col_name == 'age' or col_name == 'trestbps' or col_name == 'chol' or col_name == 'thalach':
            def lognorm(arg_x, s, loc, scale):
                return sp.stats.lognorm.pdf(arg_x, s, loc, scale)


            def lst_sqrs(par_ar, gt, x):
                s, loc, scale = par_ar
                return sum((gt - lognorm(x, s, loc, scale)) ** 2)


            lse_params = minimize(lst_sqrs, [1.1, 1.1, 1.1], method='Nelder-Mead', args=(hist_vals, hist_bins),
                                  tol=0.001, options={'disp': False})
            pdf_lsm = sp.stats.lognorm.pdf(x, *lse_params.x)

        mle_params_list.append(mle_params)
        lse_params_list.append(lse_params.x)

        axes[i, j].plot(x, pdf_lsm, color='green', label='LSE')
        axes[i, j].set_xlabel(col_names[col_idx], fontsize=15)
        axes[i, j].legend()
        col_idx += 1

        table.add_row([col_name, mle_params, lse_params.x])
        print(f'column name: {col_name}\nMLE: {mle_params}\nLSE: {lse_params.x}')
        print(' ')

plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
col_idx = 0

# Prepare percentiles (quantiles) points
percs_num = 50
percs = np.linspace(0, 100, percs_num)

for i in range(2):
    for j in range(2):
        col_name = col_names[col_idx]
        # Calculation of quantiles
        qn_real = np.percentile(df[col_name], percs)

        if col_name == 'age' or col_name == 'trestbps':
            qn_theor = sp.stats.lognorm.ppf(percs / 100.0, *mle_params_list[col_idx])
        elif col_name == 'chol' or col_name == 'thalach':
            qn_theor = sp.stats.chi2.ppf(percs / 100.0, *mle_params_list[col_idx])


        # Building a quantile biplot
        min_amount, max_amount = df[col_name].min(), df[col_name].max()

        axes[i, j].plot(qn_real, qn_theor, ls="", marker="o", markersize=6, color='red')
        axes[i, j].plot([min_amount, max_amount], [min_amount, max_amount], color="grey", ls="--")
        axes[i, j].set_xlim(min_amount, max_amount)
        axes[i, j].set_ylim(min_amount, max_amount)
        axes[i, j].set_xlabel('Empirical distribution', fontsize = 15)
        axes[i, j].set_ylabel('Theoretical distribution', fontsize = 15)
        axes[i, j].set_title(col_names[col_idx], fontsize = 18)

        col_idx += 1


plt.show()

table = PrettyTable()
table.field_names = ["column name", "type", "Kstest", "CramerVonMises"]

for col_idx in range(len(col_names)):

    col_name = col_names[col_idx]

    if col_name == 'age':
        ks = sp.stats.kstest(df[col_name], 'lognorm', lse_params_list[col_idx], N=100)
        cvm = sp.stats.cramervonmises(df[col_name], 'lognorm', lse_params_list[col_idx])

    if col_name == 'trestbps':
        ks = sp.stats.kstest(df[col_name], 'lognorm', mle_params_list[col_idx], N=100)
        cvm = sp.stats.cramervonmises(df[col_name], 'lognorm', mle_params_list[col_idx])

    elif col_name == 'chol':
        ks = sp.stats.kstest(df[col_name], 'lognorm', lse_params_list[col_idx], N=100)
        cvm = sp.stats.cramervonmises(df[col_name], 'lognorm', lse_params_list[col_idx])

    elif col_name == 'thalach':
        ks = sp.stats.kstest(df[col_name], 'lognorm', lse_params_list[col_idx], N=100)
        cvm = sp.stats.cramervonmises(df[col_name], 'lognorm', lse_params_list[col_idx])

    table.add_rows([[col_name, 'pvalue', ks[1], cvm.pvalue]])

print(table)