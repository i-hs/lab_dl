"""
MNIST 데이터를 사용한 가중치 초깃값과 신경망 성능 비교
"""
import numpy as np
import matplotlib.pyplot as plt
from ch06.ex02_sgd import Sgd
from ch06.ex05_adam import Adam
from common.multi_layer_net import MultiLayerNet

# 실험 조건 세팅
from dataset.mnist import load_mnist

weight_init_types = {
    'std=0.01': 0.01,
    'Xavier': 'sigmoid',  # 가중치 초깃값: N(0, sqrt(1/n))
    'He': 'relu'  # 가중치 초깃값: N(0, sqrt(2/n))
}

# 각 실험 조건 별로 테스트할 신경망을 생성
neural_nets = dict()
train_losses = dict()
for key, type in weight_init_types.items():
    neural_nets[key] = MultiLayerNet(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=type)
    train_losses[key] = []  # 빈 리스트 생성 - 실험(학습)하면서 손실값들을 저장

# MNIST train/test 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

iterations = 2_000  # 학습 회수
batch_size = 128  # 1번 학습에 사용할 샘플 개수(미니 배치)
optimizer = Sgd(learning_rate=0.01)  # 파라미터 최적화 알고리즘
# optimizer를 변경하면 테스트
optimizer = Adam()

np.random.seed(109)
for i in range(iterations):  # 2,000번 반복하면서
    # 미니 배치 샘플 랜덤 추출
    mask = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[mask]
    Y_batch = Y_train[mask]

    for key, net in neural_nets.items():  # 테스트 신경망 종류마다 반복
        # gradient 계산
        gradients = net.gradient(X_batch, Y_batch)
        # 파라미터(W, b) 업데이트
        optimizer.update(net.params, gradients)
        # 손실(loss) 계산 -> 리스트 추가
        loss = net.loss(X_batch, Y_batch)
        train_losses[key].append(loss)
    # 손실 일부 출력
    if i % 100 == 0:
        print(f'===== iteration #{i} =====')
        for key, loss_list in train_losses.items():
            print(key, ':', loss_list[-1])

# x축-반복 회수, y축-손실 그래프
x = np.arange(iterations)
for key, loss_list in train_losses.items():
    plt.plot(x, loss_list, label=key)
plt.title('Weight Init Compare')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()




