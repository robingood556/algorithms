import random
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as mp
from scipy.optimize import minimize
from typing import List


alpha = random.uniform(0, 1)
betha = random.uniform(0, 1)
b = np.random.normal(0, 1, 100)
print(f'alpha: {alpha}')
print(f'betha: {betha}')

df = pd.DataFrame(columns=('X', 'Y'))

print(df)
print(f'b: {b}, len: {len(b)}, b[0]: {b[0]}')

for k in range(100):
    df.loc[k] = [k / 100, alpha*(k / 100) + betha + b[k]]
print(f'alpha: {alpha}')
print(f'betha: {betha}')
print(f'df: {df}')
print(df['X'][1])

def compute_cost_for_exhaustive(x, y, w, b): 
    # number of training examples
    m = df['Y'].shape[0] 
    
    cost_sum = 0 
    for i in range(m):
        divider = (b * x[i] + 1)
        if b * x[i] == 1:
            break
        f_wb = w / divider
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  

    return cost_sum
def compute_cost(x): 
    # number of training examples
    m = df['Y'].shape[0] 
    
    cost_sum = 0 
    for i in range(m):
        f_wb = x[0] / (df['X'][i] * x[1] + 1)
        cost = (f_wb - df['Y'][i]) ** 2  
        cost_sum = cost_sum + cost  
    return cost_sum

def Nelder_Mead():
    initial_guess = np.array([0, 0])
    Nelder_Mead = minimize(compute_cost, initial_guess, method="Nelder-Mead")
    print(f'Nelder: {Nelder_Mead}')
    return Nelder_Mead.x

def exhaustive_search():
    cost_func_min = 300
    parameter_set = []
    for w in np.arange(0, 1, 0.001):
        for b in np.arange(0, 1, 0.001):
            calculate_cost = compute_cost_for_exhaustive(df["X"], df["Y"], w, b)
            print(f'cost calculatred: {calculate_cost}')
            print(f'at parameter set: {w, b}')
            if calculate_cost < cost_func_min:
                cost_func_min = calculate_cost
                parameter_set = [w, b]
    # print(f'parameter_set: {parameter_set}')
    # print(f'cost function minimum: {cost_func_min}')
    return parameter_set


def BFGS():
    initial_guess = np.array([0, 0])

    BFGS = minimize(compute_cost, initial_guess, method="BFGS")
    # print(Nelder_Mead)
    print(f'BFGS: {BFGS}')
    return BFGS.x



Nelder_Mead_res = Nelder_Mead()
Gauss_res = BFGS()
Exhaustive_res = exhaustive_search()


y_Nelder_Mead_res = Nelder_Mead_res[0] / (df['X']*Nelder_Mead_res[1] + 1)
y_Gauss_res = Gauss_res[0] / (df['X']*Gauss_res[1] + 1)
y_Exhaustive_res = Exhaustive_res[0] / (df['X'] + Exhaustive_res[1] + 1)



print(Nelder_Mead_res)
print(Gauss_res)
print(round(Nelder_Mead_res[0], 3), round(Nelder_Mead_res[1], 3))
# print(Exhaustive_res)

mp.plot(df["X"], df["Y"],label='Data')
mp.plot(df["X"], y_Nelder_Mead_res, label="Nelder_Mead")
mp.plot(df["X"], y_Gauss_res, label="Gauss")
mp.plot(df["X"], y_Exhaustive_res,label="Exhaustive_res")
mp.xlabel("X")
mp.ylabel("Y")
mp.legend()
mp.show()