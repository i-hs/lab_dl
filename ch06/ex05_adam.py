"""
파라미터 최적화 알고리즘
    Adam(Adaptive Moment estimative)
        AdaGrad + Momentum 알고리즘
        t: timestamp. 반복할 때마다 증가하는 숫자. update 메소드가 호출될 때마다 +1
        beta1:

"""
import matplotlib.pyplot as plt
import numpy as np
from ch06.ex01_matplot3d import fn_derivative, fn


class Adam:
    def __init__(self, lr=0.01):
        self.lr = lr  # 학습률
        self.t = 0
        self.m = dict()
        self.v = dict()
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.etha = 1e-8


    def update(self, params, gradients):
        self.t += 1
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        else:
            for key in params:
                self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * gradients[key]
                self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * gradients[key] * gradients[key]
                m_hat = self.m[key] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta_2 ** self.t)
                params[key] -= self.lr * m_hat / np.sqrt(v_hat + self.etha)


if __name__ == '__main__':
    adam = Adam(0.1)
    params = {'x': -7., 'y': 2.}
    gradients = {'x': 0., 'y': 0.}
    x_history = []
    y_history = []
    for i in range(100):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adam.update(params, gradients)

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
    plt.title('Adam')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()
