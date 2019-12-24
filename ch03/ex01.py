"""
perceptron:
    - 입력: (x1, x2)
    - 출력:
        a = x1 * w1 + x2 * w2 + b 계산
        y = 1, a >  임계값
          = 0, a <= 임계값
신경망의 뉴런(neuron)에서는 입력 신호의 가중치 합을 출력값으로 변환해 주는 함수가 존재
-> 활성화 함수(activation function)
"""
import math
import matplotlib.pyplot as plt
import numpy as np

def step_function(x_array):
    result_array = np.empty([1,1])
    for x_i in x_array:
        print(x_i)
        if x_i > 0:
            np.append(result_array, 1)
            print(result_array)
        else:
            np.append(result_array, 0)
    return result_array


def step_function(x):
    """
    Step Function.

    :param x: numpy.ndarray
    :return: step(계단) 함수 출력(0 또는 1)로 이루어진 numpy.ndarray
    """
    y = x > 0  # [False, False, ..., True]
    return y.astype(np.int)


def step_function(x):

    # result = []
    # for x_i in x:
    #     if x > 0:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # return np.array(result)
    y = x > 0  # [False, False, ... True]
    return y.astype(np.int)

def sigmoid(x):
    """ sigmoid = 1 / (1 + exp(-x))"""
    # return 1 / (1 + math.exp(x))
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU(Rectified Linear Unit)
        y = x, if x > 0
          = 0, otherwise
    """
    return np.maximum(0, x)

if __name__ == '__main__':
    x = np.arange(-3, 4)
    print('x = ', x)
    print(step_function(x), end ='\t')
    print()
    # sigmoid
    print('sigmoid =', sigmoid(x))
    # for x_i in x:
    #     print(sigmoid1(x_i), end = ' ')
    # print()

    # step 함수, sigmoid 함수를 하나의 그래프에 출력
    x = np.arange(-5, 5, 0.01)
    y1 = step_function(x)
    y2 = sigmoid(x)
    plt.plot(x, y1, label = 'Step_Function')
    plt.plot(x, y2, label = 'Sigmoid_Function')
    plt.legend()
    plt.show()

    # print(relu(x))
    x = np.arange(-3, 4)
    y3 = relu(x)
    plt.title('Relu')
    plt.plot(x, y3)
    plt.show()

