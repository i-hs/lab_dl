import numpy as np
from ch05.ex10_twolayer import TwoLayerNetwork
from dataset.mnist import load_mnist
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(106)
    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
    # 2층 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)

    # 100번 반복
    epochs = 30
    batch_size = 128  # 한번에 학습시키는 입력 데이터 개수
    learning_rate = 0.2

    iter_size = max(X_train.shape[0] // batch_size, 1)
    print(iter_size)

    # 전체 데이터 학습을 30회(epochs) 반복
    try_number = []
    loss_record = []
    accuracy_record = []
    for j in range(epochs):
        for i in range(iter_size):
            # forward -> backward -> gradient : gradient 함수 안에 모두 포함
            # batch_size만큼 데이터를 입력해서 gradient 계산
            X_data, Y_label = X_train[i * batch_size:(i + 1) * batch_size], Y_train[i * batch_size:(i + 1) * batch_size]
            gradients = neural_net.gradient(X_data, Y_label)
            loss = neural_net.loss(X_data, Y_label)
            accuracy = neural_net.accuracy(X_data, Y_label)
            # 가중치/편향 행렬들을 수정
            for key in gradients:
                neural_net.params[key] -= learning_rate*gradients[key]
        # 학습 데이터 세트를 무작위로 섞는 (shuffle) 코드 추가
        X_data, Y_label = shuffle(X_data, Y_label)

        try_number.append(j)
        # loss를 계산해서 출력
        print(f'loss{j}: {loss}')
        loss_record.append(loss)
        # accuracy를 계산해서 출력
        print(f'accuracy{j}: {accuracy}')
        accuracy_record.append(accuracy)

    plt.plot(try_number, loss_record)
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.show()

    plt.plot(try_number, accuracy_record)
    plt.xlabel('x')
    plt.ylabel('accuracy')
    plt.show()