import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset.mnist import load_mnist

if __name__ == '__main__':
    # 이미지 파일 오픈
    img = Image.open('desert.jpg')
    # 이미지 객체를 numpy 배열 형태(3차원 배열)로 변환
    img_pixel = np.array(img)
    print('img_pixel:', img_pixel.shape)  # (height, width, color-depth)
    # print(img_pixel)

    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False,
                                                      flatten=False)
    print('x_train', x_train.shape)  # (samples, color, height, width)
    print('x_train[0]', x_train[0].shape)  # (color, height, width)
    # plt.imshow(x_train[0])
    # (c, h, w) 형식의 이미지 데이터는 matplotlib이 사용할 수 없음
    # (h, w, c) 형식으로 변환해야 함.
    num_img = np.moveaxis(x_train[0], 0, 2)
    print(num_img.shape)  # (height, width, color)
    num_img = num_img.reshape((28, 28))  # 단색인 경우 2차원으로 변환
    plt.imshow(num_img, cmap='gray')
    plt.show()




