"""
파라미터 최적화 알고리즘 Momentum 알고리즘
v: 속도(velocity)
m: 모멘텀 상수(momentum constant)
lr: 학습률
W: 갱신할 파라미터
v = m * v - lr * dL/dW
W = W + v = W + m * v - lr * dL/dW

"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn, fn_derivative


class Momentum:
    def __init__(self, lr=0.01, m=0.9):
        self.lr = lr  # 학습률
        self.m = m  # 모멘텀 상수(속도 v에 곱해줄 상수)
        self.v = dict()  # 속도

    def update(self, params, gradients):
        if not self.v:
            for key in params:
                # 파라미터(x, y 등)와 동일한 shape 의 0으로 채워진 배열 생성
                self.v[key] = np.zeros_like(params[key])

        # 속도 v,  파라미터 params를 update하는 기능
        else:
            for key in params:
                # v = m * v - lr * dL / dW
                self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
                # W = W + v = W + m * v - lr * dL / dW
                params[key] += self.v[key]


if __name__ == '__main__':
    # Momentum 클래스의 인스턴스를 생성
    momentum = Momentum(lr=0.3)

    # 신경망에서 찾고자 하는 파라미터의 초깃값
    # 각 파라미터에 대한 변화율(gradient) 초기값
    params = {'x': -7, 'y': 2}
    gradients = {'x': 0, 'y': 0}

    # 각 파라미터들(x, y)을 갱신할 때마다 갱신된 값을 저장할 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

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
    plt.title('Momentum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()
