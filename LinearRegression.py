from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([3, 4, 2, 6, 7, 8], dtype=np.float64)


def creat_dataset(hm, variance, step=2, correlation=False):
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# plt.scatter(xs, ys)
# plt.show()


def best_fit_slope_and_intersept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squareed_error(ys_origin, ys_line):
    return sum((ys_origin - ys_line) ** 2)


def coefficient_of_determination(ys_origin, ys_line):
    y_mean_line = [mean(ys_origin) for y in ys_origin]
    squareed_error_regr = squareed_error(ys_origin, ys_line)
    squareed_error_y_mean = squareed_error(ys_origin, y_mean_line)

    return 1 - (squareed_error_regr / squareed_error_y_mean)


m, b = best_fit_slope_and_intersept(xs, ys)

print(m, b)

# plt.plot(m)
# plt.show()

regression_line = [(m * x) + b for x in xs]

predict_x = 8
predict_y = [(m * predict_x) + b]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
