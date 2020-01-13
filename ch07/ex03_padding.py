import numpy as np

if __name__ == '__main__':
    np.random.seed(113)

    # 1차원 ndarray
    x = np.arange(1, 6)
    print(x)
    x_pad = np.pad(x,  # 패딩을 넣을 배열
                   pad_width=1,  # 패딩 크기
                   mode='constant',  # 패딩에 넣을 숫자 타입
                   constant_values=0)  # 상수(constant)로 지정할 값
    print(x_pad)

    x_pad = np.pad(x,  # 패딩을 넣을 배열
                   pad_width=(2,3),  # 패딩 크기
                   mode='constant',  # 패딩에 넣을 숫자 타입
                   constant_values=0)
    print(x_pad)

    x_pad = np.pad(x, pad_width=2, mode='minimum')
    print(x_pad)

    # 2차원 ndarray
    x = np.arange(1, 10).reshape((3, 3))
    # axis=0 방향  before padding = 1
    # axis=0 방향  before padding = 2
    # axis=1 방향  before padding = 1
    # axis=2 방향  before padding = 2
    x_pad = np.pad(x, pad_width=(1,2), mode='constant', constant_values=0)
    print(x_pad)
    xpadw = np.pad(x, pad=width((1,2),(3,4)), mode='constant', constant_values=0)