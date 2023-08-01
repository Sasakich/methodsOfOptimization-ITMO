import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams["figure.figsize"] = (7, 7)


def f(arg):
    return arg[0] ** 2 + 1 * arg[1] ** 2


def grad_for_stochastic(x, position):
    h = 1e-5
    n = len(x)
    diagonal = np.zeros(n)
    diagonal[position] = 1
    matrix = np.diag(diagonal)
    return (f(x[:, np.newaxis] + h * matrix) - f(x[:, np.newaxis] - h * matrix)) / (2 * h)


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)


def stochastic_gradient_descents(list_range, dimension, eps, step):
    k = 0
    x = np.array(list_range)
    function = []
    p = []
    count = 0
    function.append(f(x))
    p.append(x)
    while True:
        position = random.randint(0, dimension - 1)
        x1 = x - step * grad_for_stochastic(x, position)
        function.append(f(x1))
        np.append(x, x1)
        if abs(f(x) - f(x1)) >= eps:
            k = k + 1
            count = count + 1
            x = x1
        else:
            print("number of gradient evaluations: ", k * 2)
            print("number of function evaluations: ", count)
            print(x1, " ", '{:f}'.format(f(x1)))
            t = np.linspace(-10, 10, 100)
            x_1, y_1 = np.meshgrid(t, t)
            x = np.array(p)
            plt.plot(x[:, 0], x[:, 1], '-o')
            plt.contour(x_1, y_1, f([x_1, y_1]), levels=sorted([f(p) for p in x]))
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(x_1, y_1, f([x_1, y_1]))
            plt.show()
            break


def main():
    # input
    list_range = [1, 2]  # input

    stochastic_gradient_descents(list_range, 2, 1e-5, 1e-2)


if __name__ == "__main__":
    main()
