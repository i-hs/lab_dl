"""
weight 행렬을 찾기 위한 경사 하강법(gradient descent)
"""

import numpy as np


def softmax(X):
    """
    1) X - 1차원: [x_1, x_2, ..., x_n]
    1) X - 2차원: [[x_11, x_12, ..., x_1n],
                   [x_21, x_22, ..., x_2n],
                   ...]
    """
    dimension = X.ndim
    if dimension == 1:
        m = np.max(X)  # 1차원 배열의 최댓값을 찾음.
        X = X - m  # 0 이하의 숫자로 변환 <- exp함수의 overflow를 방지하기 위해서.
        y = np.exp(X) / np.sum(np.exp(X))
    elif dimension == 2:
        # m = np.max(X, axis=1).reshape((len(X), 1))
        # # len(X): 2차원 리스트 X의 row의 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis=1).reshape((len(X), 1))
        # y = np.exp(X) / sum
        Xt = X.T  # X의 전치 행렬(transpose)
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T

    return y


def _cross_entropy(y_pred, y_true):
    delta = 1e-7  # log0 = -inf 가 되는 것을 방지하기 위해서 더해줄 값
    return -np.sum(y_true * np.log(y_pred + delta))


def cross_entropy(y_pred, y_true):
    if y_pred.ndim == 1:
        ce = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true) / len(y_pred)
    return ce


def partial_gradient_dim_1(fn, x):
    x = x.astype(np.float, copy=False)  # 실수 타입
    gradient = np.zeros_like(x)  # np.zeros(shape=x.shape)
    # h = 1e-4  # 0.0001
    h = 0.1
    # print(' in partial gradient dim 1')
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        # print(f'x[{i}1]= {x[i]}')
        fh1 = fn(x)
        # print('fh1 :', fh1)
        x[i] = ith_value - h
        # print(f'x[{i}2]= {x[i]}')
        fh2 = fn(x)
        # print('fh2 :', fh2)
        gradient[i] = (fh1 - fh2) / (2 * h)
        # print(f'gradient[{i}] = {gradient[i]}')
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
            # print('x_i:', x_)
            gradient[i] = partial_gradient_dim_1(fn, x_)  # 요기가 문제
            # print(f'gradient[{i}]:{gradient[i]}')
        return gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3)
        # 가중치 행렬(2x3)의 초기값 랜덤 추출
        # randn(n: normal distribution을 만족하는 숫자 랜덤 추출): shape (row, column)

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        # print('predict:', y)
        return y

    def loss(self, x, y_true):
        """손실 함수(loss function) - cross entropy
            x = training data, y_true = label (정답)
        """
        y_pred = self.predict(x)
        ce = cross_entropy(y_pred, y_true)
        # print('ce:', ce)
        return ce

    def gradient(self, x, t):
        """ x: 입력, t: 출력 실제 값(정답 레이블)"""
        fn = lambda W: self.loss(x, t)  # training data x에 대한 손실함수를 계산하여 편미분
        # print('fn:', fn)
        return partial_gradient(fn, self.W)

    def gradient_method(self, x, t, lr=0.1, step=100):
        # x = x_init  # 점진적으로 변화시킬 변수
        # W = self.W
        W_history = []  # x가 변화되는 과정을 저장할 배열
        for i in range(step):  # step 회수만큼 반복하면서
            W_history.append(self.W.copy())  # x의 복사본을 x 변화 과정에 기록
            grad = self.gradient(x, t)  # x에서의 gradient를 계산
            self.W -= lr * grad  # x_new = x_init - lr * grad: x를 변경 lr : 변화율
            # print('W:', self.W)
        return self.predict(x)


if __name__ == '__main__':
    # SimpleNetwork 클래스 객체를 생성
    network = SimpleNetwork()  # 생성자 호출  -> init method 호출
    # print('W =', network.W)

    # x = [0.6, 0.9]일 때 y_true = [0, 0, 1]이라고 가정
    x = np.array([0.6, 0.9])
    y_true = np.array([0., 0., 1.])
    # print('x =', x)
    # print('y_true =', y_true)

    y_pred = network.predict(x)
    print('y_pred =', y_pred)

    ce = network.loss(x, y_true)
    # print('cross entropy =', ce)
    #
    result = network.gradient_method(x, y_true)
    print('result =', result)

    # print('g1 =', g1)

    lr = 0.2
    # for i in range(1000):
    #     g1 = network.gradient(x, y_true)
    #     network.W -= lr * g1
    #     # print('W =', network.W)
    #     print('y_pred =', network.predict(x))
    #     print('ce =', network.loss(x, y_true))
    #     # print('lr:', lr)
    #     # print('g1:', g1)