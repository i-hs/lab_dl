"""
파라미터 최적화 알고리즘 5) RMSProp
SGD: W = W - lr * dL/dW
    SGD의 단점은 학습률(lr)을 학습 동안에 변경할 수 없다.
AdaGrad: W = W - (lr/sqrt(h)) * dL/dW
    학습하는 동안 학습률(lr)을 계속 업데이트.
    h = h + (dL/dW)**2
    학습을 계속 하다보면, 어느 순간 갱신되는 양이 0이 되는 경우가 발생할 수 있음.
    -> 더이상 학습 효과가 발생하지 않는다.
AdaGrad의 갱신량이 0이 되는 문제를 해결하기 위한 알고리즘: RMSProp
    rho: decay-rate(감쇄율)를 표현하는 하이퍼 파라미터
    h = rho * h + (1 - rho) * (dL/dW)**2
    -> h를 학습량(lr)을 변화시키는 데 사용.
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn, fn_derivative


class RMSProp:
    def __init__(self, lr=0.01, rho=0.99):
        self.lr = lr  # 학습률(learning rate)
        self.rho = rho  # decay rate
        self.h = dict()

    def update(self, params, gradients):
        if not self.h:  # h가 비어있는 딕셔너리이면,
            for key in params:
                # 파라미터의 shape과 동일하게 0으로 채워진 배열을 생성
                self.h[key] = np.zeros_like(params[key])

        epsilon = 1e-8  # 0으로 나누는 것을 방지하기 위해서
        for key in params:
            # h = rho * h + (1 - rho) * (dL/dW)**2
            self.h[key] = self.rho * self.h[key] + \
                          (1 - self.rho) * gradients[key] ** 2
            # W = W - (lr / sqrt(h)) * dL/dW
            params[key] -= (self.lr / (np.sqrt(self.h[key]) + epsilon)) * \
                           gradients[key]


if __name__ == '__main__':
    params = {'x': -7.0, 'y': 2.0}
    gradients = {'x': 0.0, 'y': 0.0}

    rmsprop = RMSProp(lr=0.1)  # learning_rate=0.01, 0.1
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        rmsprop.update(params, gradients)
        print(f"({params['x']}, {params['y']})")

    # 등고선(contour) 그래프
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RMSProp')
    plt.axis('equal')
    # x_history, y_history를 plot
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()



