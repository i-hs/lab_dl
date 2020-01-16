"""
4차원 데이터를 2차원으로 변환한 후에 max pooling 구현
"""
import numpy as np
from common.util import im2col


if __name__ == '__main__':
    np.random.seed(116)

    # 가상의 이미지 데이터(c,h,w) = (3,4,4) 1개를 난수로 생성 -> (1,3,4,4)
    x = np.random.randint(10, size=(1, 3, 4, 4))
    print(x, 'shape:', x.shape)

    # 4차원 데이터를 2차원 ndarray로 변환
    col = im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)
    print(col, 'shape:', col.shape)  # 4*12

    # max pooling : 채널별로 최댓값을 찾음
    # 채널별 최댓값을 쉽게 찾기 위해 2차원 배열의 Shape을 변환
    col = col.reshape(-1, 2 * 2)  # (-1, fh*fw)
    print(col, 'shape:', col.shape)

    # 각 행(row)에서 최댓값을 찾음.
    out = np.max(col, axis=1)
    print(out, 'shape:', out.shape)

    # 1차원 pooling의 결과를 4차원으로 변환: (n, oh, ow, c) → (n, c, oh, ow)
    out = out.reshape(1, 2, 2, 3)
    print(out)
    out = out.transpose(0, 3, 1, 2)
