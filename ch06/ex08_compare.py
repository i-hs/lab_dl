"""
파라미터 최적화 알고리즘 6개의 성능 비교 - 손실(loss), 정확도(accuracy)
"""
import matplotlib.pyplot as plt
import numpy as np

from ch05.ex10_twolayer import TwoLayerNetwork
from ch06.ex02_sgd import Sgd
from ch06.ex04_adagrad import AdaGrad
from ch06.ex05_adam import Adam
from ch06.ex06_RMSProp import RMSProp
from ch06.ex07_nesterov import Nesterov
from ch06.ex03_momentum import Momentum
from dataset.mnist import load_mnist


if __name__ == '__main__':
    # MNIST(손글씨 이미지) 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 최적화 알고리즘을 구현한 클래스의 인스턴스들을 dict에 저장
    optimizers = dict()
    optimizers['sgd'] = Sgd()
    optimizers['momentum'] = Momentum()
    optimizers['adagrad'] = AdaGrad()
    optimizers['adam'] = Adam()
    optimizers['rmsprop'] = RMSProp()
    optimizers['nesterov'] = Nesterov()

    # 은닉층 1개, 출력층 1개로 이루어진 신경망을 optimizer 개수만큼 생성
    # 각 신경망에서 손실들을 저장할 dict를 생성
    neural_nets = dict()
    train_losses = dict()
    for key in optimizers:
        neural_nets[key] = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
        train_losses[key] = []  # loss들의 이력(history)를 저장

    # 각각의 신경망을 학습시키면서 loss를 계산/기록
    iterations = 2_000  # 총 학습 회수
    batch_size = 128  # 한 번 학습에서 사용할 미니 배치 크기
    train_size = X_train.shape[0]
    # np.random.seed(108)
    for i in range(iterations):  # 2000번 학습을 반복
        # 학습 데이터, 레이블에서 미니 배치 크기만큼 랜덤하게 데이터를 선택
        batch_mask = np.random.choice(train_size, batch_size)
        # 0 ~ 59,999(train_size) 사이의 숫자중에서 128(batch_size)개의 숫자를 임의로 선택
        # 학습에 사용할 미니 배치 데이터/레이블을 선택
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]
        # 선택된 학습 데이터/레이블을 사용해서 gradient들을 계산
        for key in optimizers:
            # 각각의 최적화 작업에 대해 같은 작업을 반복
            gradients = neural_nets[key].gradient(X_batch, Y_batch)
            # 신경망이 가지고 있는 파라미터(W, b)와 위의 gradient를 optimizer으로넘김
            optimizers[key].update(neural_nets[key].params, gradients)
            # 미니 배치의 손실을 계산
            loss = neural_nets[key].loss(X_batch, Y_batch)
            train_losses[key].append(loss)
        # 100번째 학습마다 계산된 손실을 출력
        if (i+1)%100 == 0:
            print(f'===== training {i+1} =====')
            for key in optimizers:
                print(key, ':', train_losses[key][-1])

    # 각각의 최적화 알고리즘 별 손실 그래프
    x = np.arange(iterations)  # x좌표 - 학습 회수
    for key, losses in train_losses.items():
        plt.plot(x, losses, label=key)
    plt.legend()
    plt.show()