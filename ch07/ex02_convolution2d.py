"""
2차원 Convolution(합성곱) 연산
"""
import numpy as np


def convolution_2d(x, w):
    """x, w: 2d ndarray. x의 shape이 w.shape와 같다.
        x와 w의 교차 상관 연산 결과를 return
    """
    xh, xw = x.shape[0], x.shape[1]

    # 2d 배열 w의 가로(width) ww,  세로(height) wh
    wh, ww = w.shape[0], w.shape[1]
    row_num = xh - wh + 1
    col_num = xw - ww + 1
    result = []
    for i in range(row_num):
        for j in range(col_num):
            x_sub = x[i:wh + i, j:ww + j]
            fma = np.sum(x_sub * w)
            result.append(fma)
    conv_ = np.array(result).reshape((row_num, col_num))
    return conv_


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 10).reshape((3, 3))
    print(x)
    w = np.array([[2, 0],
                  [0, 0]])
    print(w)

    # 2d 배열 x의 가로(width) xw 2,  세로(height) xh 2
    xh, xw = x.shape[0], x.shape[1]

    # 2d 배열 w의 가로(width) ww 2,  세로(height) wh 2
    wh, ww = w.shape[0], w.shape[1]

    x_sub1 = x[0:wh, 0:ww]
    print('x_sub1:', x_sub1)
    fma1 = np.sum(x_sub1 * w)
    print('fma1:', fma1)
    x_sub2 = x[0:wh, 1:1 + ww]
    print('x_sub2:', x_sub2)
    fma2 = np.sum(x_sub2 * w)
    print('fam2:', fma2)

    x_sub3 = x[1:1 + wh, 0:ww]
    print('x_sub3:', x_sub3)
    fma3 = np.sum(x_sub3 * w)
    print('fam3:', fma3)

    x_sub4 = x[1:1 + wh, 1:1 + ww]
    print('x_sub4:', x_sub4)
    fma4 = np.sum(x_sub4 * w)
    print('fam4:', fma4)

    conv = np.array([fma1, fma2, fma3, fma4]).reshape((2, 2))
    print('conv:', conv)

    x_result = convolution_2d(x, w)
    print(x_result)

    x = np.random.randint(10, size=(5, 5))
    w = np.random.randint(5, size=(3, 3))
    x_result = convolution_2d(x, w)
    print('x:', x)
    print('w:', w)
    print('result:', x_result)
