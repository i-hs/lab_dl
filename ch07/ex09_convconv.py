"""
im2col 함수를 이용한 convolution 구현
"""
import numpy as np
from common.util import im2col


if __name__ == '__main__':
    np.random.seed(115)

    # p.238 그림 7-11
    # 가상의 이미지 데이터 1개
    # (number of image, color, height, width)
    x = np.random.randint(10, size=(1, 3, 7, 7))
    print(x, ', x.shape:', x.shape)

    # (3, 5, 5) 크기의 필터 1개 생성
    # (fn, c, fh, fw) = number of filter, color-dept, filter height, filter width)
    w = np.random.randint(5, size=(1, 3, 5, 5))
    print(w, ', w.shape:', w.shape)

    # 필터를 stride=1, padding=0으로 적용하면서 convolution 연산
    # 필터를 1차원으로 펼침 -> c *fh*fw = 3*5*5 = 75
    # 이미지 데이터 x를 함수 im2col에 전달
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col:', x_col.shape)\

    # 4차원 배열인 필터 w를 2차원 배열로 변환
    w_col = w.reshape(1, -1)  # row의 개수가 1, 모든 원소들은 column인 모양으로 변환
    print('w_col:', w_col.shape)

    w_col = w_col.T
    print('w_col.T:', w_col.shape)

    # 2차원으로 변환된 이미지와 필터를 행렬 dot product 연산
    out = x_col.dot(w_col)
    print('out:', out.shape)

    # dot product의 결과를 (fn, oh, ow, ?) 형태로 reshape
    out = out.reshape(1, 3, 3, -1)
    print('out:', out.shape)  # (1, 3, 3, 1) = (fn, oh, ow, c)
    out = out.transpose(0, 3, 1, 2)
    print('out:', out.shape)  # ~> (fn, c, oh, ow)

    # w.shape (10, 3, 5, 5) 생성
    x = np.random.randint(10, size=(1, 3, 7, 7))
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)

    w = np.random.randint(10, size=(10, 3, 5, 5))
    w_col = w.reshape(10, 75)
    # w를 변형: (fn, c*fh*fw)

    # x_col @ w.T 의 shape 확인
    out = x_col.dot(w_col.T)
    print('out.shape1:', out.shape)

    # dot 연산의 결과를 reshape:
    out = out.reshape(10, 3, 3, -1)
    print('out.shape2:', out.shape)

    out = out.transpose(0, 3, 1, 2)
    print('out.shape3:', out.shape)

    print('-----------------------')
    # p.239 그림 7-13, p.244 그림 7-19 참조
    # (3, 7, 7) shape의 이미지 12개를 난수로 생성 -> (n, c, h, w) = (12, 3, 7, 7)
    x = np.random.randint(10, size=(12, 3, 7, 7))

    # (3, 5, 5) shape의 필터 10개 난수로 생성 -> (fn, c, fh, ow) = (10, 3, 5, 5)
    w = np.random.randint(10, size=(10, 3, 5, 5))


    # stride=1, padding=0일 때, output height, output width =?
    # 3, 3

    # 이미지 데이터 x를 im2col 함수를 사용해서 x_col로 변환 -> shape?
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col.shape:', x_col.shape)

    # 필터 w를 x_col과 dot 연산을 할 수 있도록 reshape & transpose: w_col -> shape?
    w_col = w.reshape(10, 75)
    print('w_col.shape:', w_col.shape)
    w_col_T = w_col.T
    # x_col @ w_col
    out=x_col.dot(w_col_T)
    print('result.shape:', out.shape)
    # @ 연산의 결과를 reshape & transpose
    out = out.reshape(10, 3, 3, -1)
    out = out.transpose(0, 3, 1, 2)
    print('out.reshape', out.shape)


