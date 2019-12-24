"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle
import numpy as np

from dataset.mnist import load_mnist


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만들어둔 가중치 행렬을 읽어 옴
    with open('sample_weight.pkl', mode='rb') as file:
        network1 = pickle.load(file)
    print(network1.keys())
    # W1, W2, W3, b1, b2, b3 shape 확인
    for el in network1.keys():
        print(network1[el].shape)
    return network1


def predict(nw, xt):
    """forward 함수 : 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터에 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    # bias 행렬:
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(xt, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)
    z3 = sigmoid(a3)
    y = softmax(z3)
    return y


def accuracy(pred, test):
    """정확도 계산: 테스트 데이터의 레이블과 테스트 데이터 예측값을 파라미터에 전달받아서,
    정확도(accuracy) = (정답개수) / (전체개수)를 return"""
    p = np.argmax(pred)
    accuracy_cnt = 0

    if p == test[i]:
        accuracy_cnt += 1
    return accuracy_cnt / len(pred)

def sigmoid(x):
    """ sigmoid = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    max_x = np.max(x)
    y = np.exp(x - max_x) / np.sum(np.exp(x-max_x))
    return y


if __name__ == '__main__':
    network = init_network()
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=False)
    accuracy_cnt = 0
    for i in range(len(X_test)):
        y = predict(network, X_test[i])  # forward 함수랑 같다
        # acc = accuracy(y_pred, y_test[i])
        # print('accuracy: ', acc)
        p = np.argmax(y)
        if p == y_test[i]:
            accuracy_cnt += 1
    print('Accuracy: ', str(float(accuracy_cnt) / len(X_test)))
