import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn_derivative, fn

"""
파라미터 최적화 알고리즘 3) AdaGrad(Adaptive Gradient)
    SGD( W = W - lr * grad )에서는 학습률이 고정되어 있음.
    AdaGrad에서 학습률을 변화시키면서 파라미터를 최적화함.
    처음에는 큰 학습률로 시작, 점점 학습률을 줄여나가면서 파라미터를 갱신.
    h = h + grad * grad
    lr = lr / sqrt(h)
    W = W - (lr/sqrt(h)) * grad
"""

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr  # 학습률
        self.h = dict()

    def update(self, params, gradients):
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])
        else:
            for key in params:
                self.h[key] += gradients[key]*gradients[key]
                params[key] -= self.lr * gradients[key]/np.sqrt(self.h[key]+ (1e-8))


if __name__ == '__main__':
    adagrad = AdaGrad(1.5)
    params = {'x': -7., 'y': 2.}
    gradients = {'x': 0., 'y': 0.}
    x_history = []
    y_history = []
    for i in range(100):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adagrad.update(params, gradients)


    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    # f(x, y) 함수를 등고선으로 표현
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.title('AdaGrad')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()
