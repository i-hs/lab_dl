# from ch03.ex11 import softmax
import numpy as np

from ch04.ex08 import cross_entropy


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


def cross_entropy(y_pred, y_true):
    delta = 1e-7
    if y_pred.ndim == 1:
        return -np.sum(y_true * np.log(y_pred + delta))
    elif y_pred.ndim == 2:
        return -np.sum(y_true * np.log(y_pred + delta)) / len(y_pred)


class SoftmaxWithLoss:
    def __init__(self):
        self.y_true = None  # 정답 레이블을 저장하기 위한 field. one-hot-encoding
        self.y_pred = None  # softmax 함수의 출력(예측 레이블)을 저장하기 위한 field.
        self.loss = None

    def forward(self, X, Y_true):
        self.y_true = Y_true
        self.y_pred = softmax(X)
        self.loss = cross_entropy(self.y_pred, self.y_true)
        return self.loss

    def backward(self, dout=1):
        n = self.y_true.shape[0]  # one-hot-encoding 행렬의 row 개수
        dx = (self.y_pred - self.y_true) / n  # 오차들의 평균
        return dx


if __name__ == '__main__':
    np.random.seed(103)
    x = np.random.randint(10, size=3)
    print('x =', x)

    y_true = np.array([1., 0., 0.])  # one-hot-encoding
    print('y =', y_true)

    swl = SoftmaxWithLoss()
    loss = swl.forward(x, y_true)  # forward propagation
    print('loss =', loss)
    print('y_pred =', swl.y_pred)

    dx = swl.backward()  # back propagation
    print('dx =', dx)

    print('손실이 가장 큰 경우')
    y_true = np.array(([0, 0, 1]))
    loss = swl.forward(x, y_true)
    print('y_pred=', swl.y_pred)
    print('loss =', loss)
    print('dx =', swl.backward())


    print('손실이 가장 작은 경우')
    y_true = np.array(([0, 1, 0]))
    loss = swl.forward(x, y_true)
    print('y_pred=', swl.y_pred)
    print('loss =', loss)
    print('dx =', swl.backward())
