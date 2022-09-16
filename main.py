# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# 0 + i*0.001 = 0.001;0 + 1*0.001 =
# 3,4,5,7


import math
from sympy import *
import sympy as sp
import numpy as np

class First_task():

    def accuracy(self):
        return 0.001

    def degree_number(self, x):
        degree = 3
        return pow(x, degree)

    def module_number(self, x):
        second_number = 0.2
        res = abs(x-second_number)
        return res

    def sin_number(self, x):
        return x * np.sin(1/x)


    def check_unim(self, start_seg, end_seg, func):
        x = Symbol('x')
        first_derivative = func.diff(x)
        second_derivative = first_derivative.diff(x)
        step = self.accuracy()

        for i in np.arange(start_seg, end_seg, step):
            res = []
            res.append(lambdify(x, first_derivative)(i))
            if all(j-i > 0 for i, j in zip(res, res[1:])) == True or lambdify(x, second_derivative)(i) >= 0:
                return True
            else:
                return False

    def segment_splitting(self, start_seg, end_seg):
        const = self.accuracy()
        return int((end_seg - start_seg) / const)

    def result(self, start_seg, end_seg, func, number_task):
        left, right = [], []
        checker = self.check_unim(start_seg, end_seg, func)
        step = self.accuracy()
        iteractions = 0

        if checker == True:
            for i in np.arange(start_seg, end_seg, step):
                iteractions += 1
                res_x = start_seg + i * step
                if number_task == 1:
                    res_y = self.degree_number(res_x)
                elif number_task == 2:
                    res_y = self.module_number(res_x)
                else:
                    res_y = self.sin_number(res_x)

                left.append(res_x)
                right.append(res_y)
        else:
            return False

        print(min(right))
        print(iteractions)



q = First_task()
x = Symbol('x')
res = q.result(0, 1, x**2, 1)





# Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# 4x**3