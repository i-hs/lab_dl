"""
f(x, y, z) = (x + y)*z
x = -2, y = 5, z = -4에서의 df/dx, df/dy, df/dz의 값을
ex01에서 구현한 MultiplyLayer와 AddLayer 클래스를 이용해서 구하세요
numerical_gradient 함수에서 계산된 결과와 비교
"""
from ch05.ex01_basic_layer import *
import numpy as np
from ch04.ex05 import partial_gradient

def numerical_gradient(fn, x):
    h = 1e-4  # 0.0001
    gradient = np.zeros_like(x)
    with np.nditer(x, flags=['c_index', 'multi_index'], op_flags=['readwrite']) as it:
        while not it.finished:
            i = it.multi_index
            ith_value = it[0]  # 원본 데이터를 임시 변수에 저장
            it[0] = ith_value + h  # 원본 값을 h만큼 증가
            fh1 = fn(x)  # f(x + h)
            it[0] = ith_value - h  # 원본 값을 h만큼 감소
            fh2 = fn(x)  # f(x - h)
            gradient[i] = (fh1 - fh2) / (2 * h)
            it[0] = ith_value  # 가중치 행렬의 원소를 원본값으로 복원.
            it.iternext()
    return gradient


def fnn(a, b, c):
    # print(array_3_var)
    return (a + b) *c


if __name__ == '__main__':
    fst_node = AddLayer()
    x, y, z = -2, 5, -4
    fst_fwd = fst_node.forward(x, y)
    scnd_node = MultiplyLayer()
    scnd_fwd = scnd_node.forward(fst_fwd, z)

    df_dfirst, df_dz = scnd_node.backward(1)
    print('df/dz:', df_dz)
    dx, dy = fst_node.backward(df_dfirst)
    print('df/dx:', dx)
    print('df/dy:', dy)

    # coordinates = np.array([-2, 5, -4])
    # num_grad_result = partial_gradient(fnn, coordinates)
    # numerical_gradient_result = numerical_gradient(fnn, coordinates)
    # print(num_grad_result)
    # print(numerical_gradient_result)

    h = 1e-12
    dx = (fnn((-2+ h), 5, -4) - fnn((-2- h), 5, -4))/ (2*h)
    print('dx`:', dx)