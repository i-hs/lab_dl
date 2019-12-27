"""
Machine Learning(기계 학습) -> Deep Learning(심층 학습)
training data set(학습 세트)/test data set(검증 세트)
신경망 층들을 지나갈 때 사용되는 가중치(Weight) 행렬, 편향 행렬(Bias Matrix)들을 찾는 것이 목적.
오차를 최소화하는 가중치 행렬들을 찾음.
손실함수/비용(Loss Function/Cost)의 값을 최소화하는 가중치 행렬을 찾음.
손실함수:
    - 평균 제곱 오차(MSE: Mean Squared Error)
    - 교차 엔트로피(Cross-Entropy
"""
import math

from dataset.mnist import load_mnist
import numpy as np

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist()

    # 10개 테스트 이미지들의 실제 값
    print('y_true:', y_true[:10])

    # 10개 테스트 이미지들의 예측 값
    y_pred = np.array([7, 2, 1, 6, 4, 1, 4, 9, 6, 9])
    print('y_pred:', y_pred)

    # 오차
    error = y_pred - y_true[:10]
    print(error)

    # 오차 제곱(squared error)
    sq_err = error ** 2
    print('squared error:', sq_err)

    # 평균 제곱 오차(mean squared error)
    mse = np.mean(sq_err)
    print('MSE:', mse)

    # RMS(Root Mean Squared_Error)
    print('RMSE:', np.sqrt(mse))
