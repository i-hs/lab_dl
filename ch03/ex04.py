import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2])

# a = x @ W + b
W1 = np.array([[1, 2, 3],
               [4, 5, 6]])
b1 = np.array([1, 2, 3])
a1 = x.dot(W1)+b1
print(a1)

# 출력 a1에 활성화 함수를 sigmoid 함수로 적용
# z1 = sigmoid(a1)

def sigmoid(x):
    """ sigmoid = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))

z1 = np.array(sigmoid(a1))
print(f'z(1) = {z1}')


# 두 번째 은닉층에 대한 가중치 행렬 W2와 bias 행렬 b2를 작성
W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
a2 = z1.dot(W2) + b2

# a2에 활성화 함수(sigmoid)를 적용
z2 = sigmoid(a2)
print(f'z(2) = {z2}')