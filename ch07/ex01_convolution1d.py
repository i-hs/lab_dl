"""
1차원 Convolution(합성곱), Cross-Correlation(교차-상관) 연산
"""
import numpy as np


def convolution_1d(x, w):
    """x, w: 1d ndarray, len(x) > len(w) . return 결과"""
    w_r = np.flip(w)
    # conv_1d = []
    # len_rslt = len(x) - len(w) + 1
    # for i in range(len_rslt):
    #     x_sub = x[i:i + len(w)]  # (0,1) (1,2) (2,3) (3,4)
    #     fma = np.sum(x_sub * w_r)
    #     conv_1d.append(fma)
    conv_1d = cross_correlation_1d(x, w_r)
    return conv_1d


def cross_correlation_1d(x, w):
    """x, w : ndarray, len(x) >-= len(w)
    x와 w의 교차 상관(cross-correlation) 연산 결과를 리턴
    -- > convolution_1d() 함수 cross_correlation_1d()를 사용하도록 수정"""
    w_r = np.flip(w)
    conv_1d = []
    len_rslt = len(x) - len(w) + 1
    for i in range(len_rslt):
        x_sub = x[i:i + len(w)]  # (0,1) (1,2) (2,3) (3,4)
        fma = np.sum(x_sub * w_r)
        conv_1d.append(fma)
    conv_1d = np.array(conv_1d)
    return conv_1d

if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print('x =', x)
    w = np.array([2, 1])
    print('w =', w)

    # Convolution 합성곱 연산
    # 1) w를 반전
    w_r = np.flip(w)  # 행렬을 반전해주는 method
    print('w_r =', w_r)

    # 2) FMA(Fused Multiply-Add)
    conv = []
    for i in range(4):
        x_sub = x[i:i + 2]  # (0,1) (1,2) (2,3) (3,4)
        fma = np.sum(x_sub * w_r)
        conv.append(fma)
    conv = np.array(conv)
    print(conv)

    # 1차원 convolution 연산 결과의 크기(원소의 개수)
    # 원소의 개수 = len(x) - len(w) + 1
    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    conv_rslt_1d = convolution_1d(x, w)
    print(conv_rslt_1d)

    # 교차 상관(Cross-Correlation) 연산
    # 합성곱 연산과 다른 점은 w를 반전시키지 않는다는 것.
    # CNN(Convolutional Neural Network, 합성곱 연산망)에서는 대부분 교차 상관을 사용
    # 가중치 행렬을 난수로 생성한 후 Gradient Descent 등을 이용해서 갱신하기 때문에,
    # 대부분의 경우 합성곱 연산 대신 교차 상관을 사용함

