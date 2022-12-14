import itertools

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.optimize import linear_sum_assignment
import time
import pandas as pd
import matplotlib.pyplot as mp
from typing import NamedTuple
from collections import namedtuple
import time


class TaskAssignment:


    def __init__(self, task_matrix):
        self.task_matrix = task_matrix
        self.min_cost, self.best_solution = self.Hungary_method(task_matrix)




    def Hungary_method(self, task_matrix):
        b = task_matrix.copy()

        for i in range(len(b)):
            row_min = np.min(b[i])
            for j in range(len(b[i])):
                b[i][j] -= row_min
        for i in range(len(b[0])):
            col_min = np.min(b[:, i])
            for j in range(len(b)):
                b[j][i] -= col_min
        line_count = 0

        while (line_count < len(b)):
            line_count = 0
            row_zero_count = []
            col_zero_count = []
            for i in range(len(b)):
                row_zero_count.append(np.sum(b[i] == 0))
            for i in range(len(b[0])):
                col_zero_count.append((np.sum(b[:, i] == 0)))

            line_order = []
            row_or_col = []
            for i in range(len(b[0]), 0, -1):
                while (i in row_zero_count):
                    line_order.append(row_zero_count.index(i))
                    row_or_col.append(0)
                    row_zero_count[row_zero_count.index(i)] = 0
                while (i in col_zero_count):
                    line_order.append(col_zero_count.index(i))
                    row_or_col.append(1)
                    col_zero_count[col_zero_count.index(i)] = 0

            delete_count_of_row = []
            delete_count_of_rol = []
            row_and_col = [i for i in range(len(b))]
            for i in range(len(line_order)):
                if row_or_col[i] == 0:
                    delete_count_of_row.append(line_order[i])
                else:
                    delete_count_of_rol.append(line_order[i])
                c = np.delete(b, delete_count_of_row, axis=0)
                c = np.delete(c, delete_count_of_rol, axis=1)
                line_count = len(delete_count_of_row) + len(delete_count_of_rol)

                if line_count == len(b):
                    break
                if 0 not in c:
                    row_sub = list(set(row_and_col) - set(delete_count_of_row))
                    min_value = np.min(c)
                    for i in row_sub:
                        b[i] = b[i] - min_value
                    for i in delete_count_of_rol:
                        b[:, i] = b[:, i] + min_value
                    break
        row_ind, col_ind = linear_sum_assignment(b)
        min_cost = task_matrix[row_ind, col_ind].sum()
        best_solution = list(task_matrix[row_ind, col_ind])
        return min_cost, best_solution

rd = random.RandomState(10000)
time_result = []
number_matrix = []
for x in range(1,500):
        task_matrix = rd.randint(0, 100, size=(x, x))
        print(task_matrix)
        ass_by_Hun = TaskAssignment(task_matrix)
        start_time = time.time_ns()
        print(start_time)
        ass_by_Hun.best_solution
        finish_time = time.time_ns()
        res = finish_time - start_time
        number_matrix.append(x)
        time_result.append(res)

model = np.poly1d(np.polyfit(number_matrix,time_result, 3))
number_matrix = np.array(number_matrix)
coeffs = np.polyfit(number_matrix,time_result, 3)
fit = np.poly1d(coeffs)

fit_curve = coeffs[0] * number_matrix**3 + coeffs[1] * number_matrix**2 + coeffs[2] * number_matrix + coeffs[3]
plt.plot(number_matrix, time_result)
plt.plot(fit_curve)
plt.xlabel("Matrix x*x")
plt.ylabel("Time in seconds")
plt.show()