# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# 0 + i*0.001 = 0.001;0 + 1*0.001 =
# 3,4,5,7


from sympy import *
import numpy as np

class First_task():

    def accuracy(self):
        return 0.001

    def degree_number(self, x):
        degree = 3
        return pow(x, degree)

    def module_number(self, x):
        second_number = 0.2
        return abs(x-second_number)

    def sin_number(self, x):
        return x * np.sin(1/x)


    def check_unim(self, arr, func):
        first_item = arr[0]
        last_item = arr[-1]
        x = Symbol('x')
        first_derivative = func.diff(x)
        second_derivative = first_derivative.diff(x)
        step = self.accuracy()

        for i in np.arange(first_item, last_item, step):
            res = []
            res.append(lambdify(x, first_derivative)(i))
            if all(j-i > 0 for i, j in zip(res, res[1:])) == True or lambdify(x, second_derivative)(i) >= 0:
                return True
            else:
                return False

    def result_num(self, arr, func, task):
        left, right = [], []
        first_item = arr[0]
        last_item = arr[-1]
        checker = self.check_unim(arr, func)
        step = self.accuracy()
        iteractions = 0

        if task == "degree":
            iteraction_func = self.degree_number
        elif task == "sin":
            iteraction_func = self.sin_number
        elif task == "module":
            iteraction_func = self.module_number
        else:
            return False

        if checker == True:
            for i in np.arange(first_item, last_item, step):
                iteractions += 1
                res_x = arr[0] + i * step

                res_y = iteraction_func(res_x)

                left.append(res_x)
                right.append(res_y)

            print(f"min : {min(right)} and iteractions: {iteractions}")
        else:
            return False

    def result_dioch(self, arr, func, task):
        eps = self.accuracy()
        first_item = arr[0]
        last_item = arr[-1]
        checker = self.check_unim(arr, func)
        midpoint = eps / 2
        iteractions = 0

        if task == "degree":
            iteraction_func = self.degree_number
        elif task == "sin":
            iteraction_func = self.sin_number
        elif task == "module":
            iteraction_func = self.module_number
        else:
            return False

        if checker == True:

            while abs(last_item - first_item) / 2 > eps:
                iteractions += 1

                x1 = (first_item + last_item - midpoint) / 2
                x2 = (first_item + last_item + midpoint) / 2

                if iteraction_func(x1) <= iteraction_func(x2):
                    last_item = x2
                else:
                    first_item = x1

            x_mid = (first_item + last_item) / 2
            func_x = iteraction_func(x_mid)

            print(x_mid, func_x, iteractions)

        else:
            return False


    def result_golden(self, arr, func, task):
        iteractions = 0
        eps = self.accuracy()
        first_item = arr[0]
        last_item = arr[-1]
        checker = self.check_unim(arr, func)

        if task == "degree":
            iteraction_func = self.degree_number
        elif task == "sin":
            iteraction_func = self.sin_number
        elif task == "module":
            iteraction_func = self.module_number
        else:
            return False

        if checker == True:

            while 0.61*abs(last_item - first_item) > eps:

                iteractions += 1

                x1 = first_item + 0.38*(last_item - first_item)
                x2 = first_item + 0.61*(last_item - first_item)

                if iteraction_func(x1) <= iteraction_func(x2):
                    last_item = x2
                else:
                    first_item = x1

            x_mid = (first_item + last_item) / 2
            func_x = iteraction_func(x_mid)

            print(x_mid, func_x, iteractions)

        else:
            return False


q = First_task()
x = Symbol('x')
res_dio = q.result_dioch([0, 1], x**3, "degree")
res_num = q.result_num([0, 1], x**3, "degree")
res_gold = q.result_golden([0, 1], x**3, "degree")