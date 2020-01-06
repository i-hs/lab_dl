"""
오차 역전파를 이용한 2층 신경망(1 은닉층 + 1 출력층)
"""
from collections import OrderedDict
import numpy as np

from ch05.ex05_relu import Relu
from ch05.ex07_affine import Affine
from ch05.ex08_softmax_loss import SoftmaxWithLoss
from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """신경망의 구조(모양) 결정"""
        np.random.seed(106)

        # 가중치 / 편향 행렬들을 초기화
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # layer 생성/초기화
        self.layers = OrderedDict()
        # 딕셔너리에 데이터가 추가된 순서가 유지되는 딕셔너리
        self.layers['affine1'] = Affine(self.params['W1'], self.params['b1'])  # Affine
        self.layers['relu'] = Relu()  # Relu
        self.layers['affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, X):
        # Y = self.layers['affine1'].forward(X)
        # Y = self.layers['relu'].forward(Y)
        # Y = self.layers['affine2'].forward(Y)
        # 위 3개의 식과 아래의 식은 같다
        # orderedDict이므로 저장한 순서대로 values가 출력된다.
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def loss(self, X, y_true):
        """입력 데이터 X와 실제 값(레이블) Y_true가 주어졌을 때,
        손실(cross-entropy)를 계산해서 리턴"""
        # 출력층(SoftmaxWithLoss) 전까지의 forward propagation을 계산
        Y_pred = self.predict(X)
        # 출력층 바로 전 계층의 출력값을 입력으로 사용해서 손실(cross-entropy)을 계산
        loss = self.last_layer.forward(Y_pred, y_true)
        return loss

    def accuracy(self, X, Y_true):
        """입력 데이터 X와 실제 값(레이블) Y_true가 주어졌을 때,
            예측 값들의 정확도를 계산해서 return"""
        Y_pred = self.predict(X)
        predictions = np.argmax(Y_pred, axis=1)
        trues = np.argmax(Y_true, axis=1)
        result = trues == predictions  # 정답과 예측값의 비교(bool) 결과를 저장한 배열
        return np.mean(result)  # True = 1, False = 0 으로 대체된 후 평균 계산됨.
        # (1 + 1 + ... + 0 + ...) / 전체 개수

    def gradient(self, X, Y_true):
        """입력 데이터 X와 실제 값(레이블) Y_true가 주어졌을 때,
            모든 레이어에 대해서 forward propagation을 수행한 후,
            오차 역전파 방법을 이용해서 dW1, db1, dW2, db2를 계산하고 리턴."""
        # 가중치/ 편향 행렬에 대한 gardients를 저장할 dictionary
        gradients = dict()

        loss = self.loss(X, Y_true)  # forward propagation

        # back propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())  # [affine1, relu, affine2] 순서로 저장됨
        layers.reverse()  # 리스트를 역순으로 바꿔주는 함수 [ affine2, relu, affine1 ]
        for layer in layers:
            dout = layer.backward(dout)

        # 모든 레이어에 대해서 역전파가 끝나면, 가중치/편향 행렬의 gradient를 찾을 수 있다.
        gradients['W1'] = self.layers['affine1'].dW
        gradients['b1'] = self.layers['affine1'].db
        gradients['W2'] = self.layers['affine2'].dW
        gradients['b2'] = self.layers['affine2'].db

        return gradients


if __name__ == '__main__':
    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
    # 데이터 shape 확인
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # 신경망 객체 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)

    for key in neural_net.params:
        print(key, ':', neural_net.params[key].shape)

    for key in neural_net.layers:
        print(key, ':', neural_net.layers[key])
    print(neural_net.last_layer)

    X = neural_net.predict(X_train)
    print(X)

    # predict 메소드 테스트
    Y_pred = neural_net.predict(X_train[0])
    print('X_train[0]:', Y_pred)
    print(np.argmax(Y_pred))
    Y_loss = neural_net.loss(X_train[0], Y_train[0])
    print('Y_loss:', Y_loss)

    Y_pred = neural_net.predict(X_train[:3])
    print('X_train[:3]:', Y_pred)
    print(np.argmax(Y_pred, axis=1))
    Y_loss = neural_net.loss(X_train[:3], Y_train[:3])
    print('Y_loss:', Y_loss)

    print('\n====accuracy method test====\n')
    # accuracy 메소드 테스트
    print(Y_train[:10])
    print('Accuracy:', neural_net.accuracy(X_train[:10], Y_train[:10]))

    print('\n==== gradient 메소드 테스트 ====\n')

    # gradient 메소드 테스트
    gradients = neural_net.gradient(X_train[:3], Y_train[:3])
    for key in gradients:
        print(gradients[key], end=' ')

