"""
배치 정규화(Batch Normalization)
신경망의 각 층에 미니 배치(mini-batch)를 전달할 때마다 정규화(normalization)을
실행하도록 강제하는 방법.
-> 학습 속도 개선 - p.213 그림 6-18
-> 파라미터(W, b)의 초깃값에 크게 의존하지 않음. - p.214 그림 6-19
-> 과적합(overfitting)을 억제.

y = gamma * x + beta
gamma 파라미터: 정규화된 미니 배치를 scale-up/down
beta 파라미터: 정규화된 미니 배치를 이동(bias)
배치 정규화를 사용할 때는 gamma와 beta를 초깃값을 설정을 하고,
학습을 시키면서 계속 갱신(업데이트)함.
"""
# p.213 그림 6-18을 그리세요.
# Batch Normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교
import numpy as np
import matplotlib.pyplot as plt
from ch06.ex02_sgd import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Momentum, AdaGrad, Adam
from dataset.mnist import load_mnist

# 배치 정규화를 사용하는 신경망
bn_neural_net = MultiLayerNetExtend(input_size=784,
                                    hidden_size_list=[100, 100, 100, 100, 100],
                                    output_size=10,
                                    weight_init_std=0.01,
                                    use_batchnorm=True)
# 배치 정규화를 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size=784,
                                 hidden_size_list=[100, 100, 100, 100, 100],
                                 output_size=10,
                                 weight_init_std=0.01,
                                 use_batchnorm=False)

# 미니 배치를 20번 학습시키면서, 두 신경망에서 정확도(accuracy)를 기록
# -> 그래프
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
# 학습 시간을 줄이기 위해 학습 데이터의 개수를 줄임.
X_train = X_train[:1000]
Y_train = Y_train[:1000]

train_size = X_train.shape[0]
batch_size = 128
learning_rate = 0.1
iterations = 20

train_accuracies = []  # 배치 정규화를 사용하지 않는 신경망의 정확도를 기록
bn_train_accuracies = []  # 배치 정규화를 사용하는 신경망의 정확도를 기록

optimizer = Sgd(learning_rate)
bn_optimizer = Sgd(learning_rate)

# 학습하면서 정확도의 변화를 기록
np.random.seed(110)

for i in range(iterations):
    # 미니 배치를 랜덤하게 선택(0~999 숫자들 중 128개를 랜덤하게 선택)
    mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[mask]
    y_batch = Y_train[mask]

    # 각 신경망에서 gradient를 계산
    gradients = neural_net.gradient(x_batch, y_batch)
    # 파라미터 업데이터(갱신) - W(가중치), b(편향)을 업데이트
    optimizer.update(neural_net.params, gradients)
    # 업데이트된 파라미터들을 사용해서 배치 데이터의 정확도 계산
    acc = neural_net.accuracy(x_batch, y_batch)
    # 정확도를 기록
    train_accuracies.append(acc)

    # 배치 정규화를 사용하는 신경망에서 같은 작업을 수행
    bn_gradients = bn_neural_net.gradient(x_batch, y_batch)
    bn_optimizer.update(bn_neural_net.params, bn_gradients)  # W(가중치), b(편향)을 업데이트
    bn_acc = bn_neural_net.accuracy(x_batch, y_batch)  # 정확도 계산
    bn_train_accuracies.append(bn_acc)  # 정확도 기록

    print(f'iteration #{i}: without= {acc}, with={bn_acc}')

# 정확도 비교 그래프
x = np.arange(iterations)
plt.plot(x, train_accuracies, label='without BN')
plt.plot(x, bn_train_accuracies, label='using BN')
plt.legend()
plt.show()

