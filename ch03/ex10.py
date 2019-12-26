import numpy as np
from ch03.ex05 import softmax

if __name__ == '__main__':



    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    s = softmax(X)
    print(s)

    # NumPy Broadcast(브로드캐스트)
    # NumPy array의 축(axis)
    #  axis=0: row의 인덱스가 증가하는 축
    #  axis=1: column의 인덱스가 증가하는 축

    # array과 scalar 간의 브로드캐스트
    x = np.array([1, 2, 3])
    print('x =', x)
    print('x + 10 =', x + 10)
    print('===========')
    # 2차원 array와 1차원 array 간의 브로드캐스트
    # (n, m) array와 (m,) 1차원 array는 브로드캐스트가 가능
    # (n, m) array와 (n,) 1차원 array인 경우는,
    # 1차원 array를 (n,1) shape으로 reshape를 하면 브로드캐스트가 가능.
    X = np.arange(6).reshape((2, 3))
    print('X shape:', X.shape)
    print('X =', X)

    a = np.arange(1, 4)
    print('a shape:', a.shape)
    print('a =', a)

    print('X + a =', X + a)

    b = np.array([10, 20])
    print('b shape:', b.shape)
    b = b.reshape((2, 1))
    print('b shape:', b.shape)
    print('X + b =', X + b)

    np.random.seed(2020)
    X = np.random.randint(10, size=(2, 3))
    print(X)
    # 1. X의 모든 원소들 중 최댓값(m)을 찾아서,
    # X - m을 계산해서 출력
    m = np.max(X)
    print(f'm = {m}')
    result = X - m  # (2,3) shape의 2차원 배열과 스칼라 간의 broadcast
    print(result)

    # 2. X의 axis=0 방향의 최댓값들(각 컬럼에서의 최댓값들)을 찾아서,
    # X의 각 원소에서, 그 원소가 속한 컬럼의 최댓값을 뺀 행렬을 출력
    m = np.max(X, axis=0)
    print(f'm = {m}, shape: {m.shape}')
    result = X - m
    print(result)

    # 3. X의 axis=1 방향의 최댓값들(각 row에서의 최댓값들)을 찾아서,
    # X의 각 원소에서, 그 원소가 속한 row의 최댓값을 뺀 행렬을 출력
    m = np.max(X, axis=1)
    print(f'm = {m}, shape: {m.shape}')
    result = X - m.reshape((2, 1))
    print(result)

    X_t = X.T  # transpose: 행렬의 행과 열을 바꾼 행렬
    print(X_t)
    m = np.max(X_t, axis=0)  # 전치 행렬에서 axis=0 방향으로 최댓값 찾음.
    result = X_t - m  # 전치 행렬에서 최댓값들을 뺌.
    result = result.T  # 전치 행렬을 다시 한번 transpose
    print(result)

    # 4. X의 각 원소에서, 그 원소가 속한 컬럼의 최댓값을 뺀 행렬의
    # 컬럼별 원소들의 합계
    m = np.max(X, axis=0)
    result = X - m
    s = np.sum(result, axis=0)
    print(s)

    # 5. X의 각 원소에서, 그 원소가 속한 row의 최댓값을 뺀 행렬의
    # row별 원소들의 합계
    m = np.max(X, axis=1)
    result = X - m.reshape((2, 1))
    s = np.sum(result, axis=1)
    print(s)

    # 표준화: 평균 mu, 표준편차 sigma -> (x - mu)/sigma -> axis = 0 방향으로
    print(X)
    X_new = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    print(X_new)

    # 표준화: 평균 mu, 표준편차 sigma -> (x - mu)/sigma -> axis = 1 방향으로
    print(X)
    X_new = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    print(X_new)





