"""
6.2 가중치 초깃값: Y = X @ W + b
신경망의 파라미터인 가중치 행렬(W)를 처음에 어떻게 초기화를 하느냐에 따라서
신경망의 학습 성능이 달라질 수 있다.
Weight의 초깃값을 모두 0으로 하면(또는 모두 균일한 값으로 하면) 학습이 이루어지지 않음.
그래서 Weight의 초깃값은 보통 정규 분포를 따르는 난수를 랜덤하게 추출해서 만듦.
그런데, 정규 분포의 표준 편차에 따라서 학습의 성능이 달라짐.
"""
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


if __name__ == '__main__':
    # 은닉층(hidden layer)에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-10, 10, 1000)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    plt.plot(x, y_sig, label='Sigmoid')
    plt.plot(x, y_tanh, label='Hyperbolic tangent')
    plt.plot(x, y_relu, label='ReLU')
    plt.legend()
    plt.title('Activation Functions')
    plt.ylim((-1.5, 1.5))
    plt.axvline()
    plt.axhline()
    plt.show()

    # 가상의 신경망에서 사용할 테스트 데이터(mini-batch)를 생성
    np.random.seed(108)
    x = np.random.randn(1000, 100)  # 정규화가 된 테스트 데이터

    node_num = 100  # 은닉층의 노드(뉴런) 개수
    hidden_layer_size = 5  # 은닉층의 개수
    activations = dict()  # 데이터가 은닉층을 지났을 때 출력되는 값을 저장

    # x dot w
    # 은닉층에서 사용하는 가중치 행렬
    w = np.random.randn(node_num, node_num)

    # a = x dot w
    a = x.dot(w)
    z = sigmoid(a)
    activations[0] = z




