import numpy as np
import matplotlib.pyplot as plt

def partial_gradient_dim_1(fn, x):
    x = x.astype(np.float)  # 실수 타입
    gradient = np.zeros_like(x)  # np.zeros(shape=x.shape)
    h = 1e-4  # 0.0001
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        fh1 = fn(x)
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2 * h)
        x[i] = ith_value
    return gradient

def partial_gradient(fn, x):
    """x = [
        [x11, x12, x13 ..],
        [x21, x22, x23 ..],
        ..
    ]
    """
    if x.ndim == 1:
        return partial_gradient_dim_1(fn, x)

    else:
        gradient = np.zeros_like(x)
        for i, x_ in enumerate(x):
            gradient[i] = partial_gradient_dim_1(fn, x_)
        return gradient




def numerical_diff(fn, x):
    """ Numerical Differential
    함수 fn과 점 x가 주어졌을 때, x에서의 함수 fn의 미분(도함수) 값"""
    h = 1e-4  # 0.0001
    return (fn(x + h) - fn(x - h)) / (2 * h)


def fn(x):
    if x.ndim ==1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


x0 = np.arange(-1, 2)
x1 = np.arange(-1, 2)
print('x0 :', x0)
print('x1 :', x1)

X, Y = np.meshgrid(x0, x1)
print('X =', X)
print('Y =', Y)

X = X.flatten()
Y = Y.flatten()
print('X =', X)
print('Y =', Y)
XY = np.array([X, Y])
print('XY =', XY)

gradients = partial_gradient(fn, XY)
print('gradients =', gradients)

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)

X, Y = np.meshgrid(x0, x1)
X = X.flatten()
Y = Y.flatten()
XY = np.array([X, Y])
gradients = partial_gradient(fn, XY)

plt.quiver(X, Y, -gradients[0], -gradients[1], angles='xy')  # X좌표, Y좌표, X좌표 미분값, Y좌표 미분값
plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.axhline(color='y')
plt.axvline(color='y')
plt.show()
