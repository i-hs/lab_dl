import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common.util import im2col
from dataset.mnist import load_mnist


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # weight - filter
        self.b = b  # bias
        self.stride = stride
        self.pad = pad
        # 중간 데이터: forward에서 생성되는 데이터 -> backward에서 사용
        self.x = None
        self.x_col = None
        self.W_col = None
        # gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        """x: 4차원 이미지 (mini-batch) 데이터"""
        self.x = x
        n, c, h, w = self.x.shape
        fn, c, fh, fw = self.W.shape
        oh = (h - fh + 2 * self.pad) // self.stride + 1  # output height
        ow = (w - fw + 2 * self.pad) // self.stride + 1  # output width

        self.x_col = im2col(self.x, fh, fw, self.stride, self.pad)
        self.W_col = self.W.reshape(fn, -1).T
        # W(fn,c,fh,fw) --> W_col(fn, c*fh*fw) --> (c*fh*fw, fn)

        out = np.dot(self.x_col, self.W_col) + self.b
        # self.x_col.dot(self.W_col)

        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        return out


if __name__ == '__main__':
    # Convolution을 생성
    W = np.zeros((1, 1, 4, 4), dtype=np.uint8)  # (filter 개수, color 수, fh, fw)  # dtype: 8bit 부호없는 정수
    W[0, 0, 1, :] = 1
    b = np.zeros(1)
    conv = Convolution(W, b)  # Convolution class의 생성자 호출
    # MNIST 데이터를 forward
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)
    # 다운로드 받은 이미지 파일을 ndarray로 변환해서 forward
    input = x_train[0:1]  # 4차원 데이터를 뽑아내려면 slicing을 사용!
    print('input:', input.shape)

    out = conv.forward(input)
    print('out:', out.shape)
    img = out.squeeze()  # 차원의 원소가 1개이면 해당 차원을 지운다
    print('img:', img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()

    img = Image.open('pengsoo.jpg')

    img_pixel = np.array(img)

    print(img_pixel.shape)  # (1357, 1920, 3)
