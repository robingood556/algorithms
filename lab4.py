import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from scipy.optimize import minimize, least_squares, differential_evolution, dual_annealing
import plotly.express as px

N_k = np.random.normal(0, 1, 1000)


df = pd.DataFrame(columns=('X', 'Y'))
for k in range(1000):

    x_k = 3*k / (1000)
    fx_k = 1 / (x_k**2 - 3*x_k + 2)

    if fx_k < -100:
        y_k = 100 + N_k[k]
    elif abs(fx_k) <= 100 and abs(fx_k) >= -100:
        y_k = fx_k + N_k[k]
    else:
        y_k = 100 + N_k[k]

    df.loc[k] = [x_k, y_k]

print(f'df: {df}')

def compute_cost(x):
    # number of training examples
    m = df['Y'].shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = (x[0] * df['X'][i] + x[1]) / (df['X'][i] ** 2 + x[2] * df['X'][i] +x[3])
        cost = (f_wb - df['Y'][i]) ** 2
        cost_sum = cost_sum + cost

    return cost_sum

def computing(x):
    # number of training examples
    res = []
    m = df['Y'].shape[0]
    for i in range(m):
        res.append(((x[0] * df['X'][i] + x[1]) / (df['X'][i] ** 2 + x[2] * df['X'][i] + x[3]) - df['Y'][i]) ** 2)
    return res

def Nelder_Mead():
    initial_guess = np.array([0.1, 0.1, 0.1, 0.1])

    Nelder_Mead = minimize(compute_cost, initial_guess, method="Nelder-Mead", options={'xatol': 1e-3})
    print(Nelder_Mead.x, Nelder_Mead.nit)
    print(Nelder_Mead)
    return Nelder_Mead.x

def Levenberg():
    initial_guess = np.array([0.1, 0.1, 0.1, 0.1])
    Levenberg = least_squares(computing, initial_guess, method="lm", xtol=1e-3)
    print(Levenberg.x)
    return Levenberg.x

def diff_un():
    diff_un = differential_evolution(compute_cost, ((-4, 4), (-4, 4), (-4, 4), (-4, 4)))
    print(diff_un.x, diff_un.nit)
    return diff_un.x

def sim_annel():
    sim_annel = dual_annealing(compute_cost, ((-4, 4), (-4, 4), (-4, 4), (-4, 4)))
    print(sim_annel)
    return sim_annel.x


Nelder_Mead_res = Nelder_Mead()
Levenberg_res = Levenberg()
diff_un_res = diff_un()
sim_annel_res = sim_annel()

y_Nelder_Mead_res = Nelder_Mead_res[0] * df['X'] + Nelder_Mead_res[1]
y_Levenberg_res = Levenberg_res[0] * df['X']  + Levenberg_res[1]
diff_un_res = diff_un_res[0] * df['X']  + diff_un_res[1]
sim_annel_res = sim_annel_res[0] * df['X'] + sim_annel_res[1]
mp.plot(df["X"], df["Y"], label='Data')
mp.plot(df["X"], y_Nelder_Mead_res, label="Nelder_Mead")
mp.plot(df["X"], y_Levenberg_res, label="Levenberg-Marquardt")
mp.plot(df["X"], diff_un_res, label="differential_evolution")
mp.plot(df["X"], sim_annel_res, label="Simulated Annealing")
mp.xlabel("X")
mp.ylabel("Y")
mp.legend()
mp.show()