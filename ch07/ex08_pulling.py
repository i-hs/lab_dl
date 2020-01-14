import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset.mnist import load_mnist


def pooling1d(x, pool_size, stride=1):
    n = x.shape[0]  # len(x)
    result_size = (n - pool_size) // stride + 1
    result = np.zeros(result_size)
    for i in range(result_size):
        x_sub = x[(i * stride):(i * stride) + pool_size]
        result[i] = np.max(x_sub)
    return result


def pooling2d(x, pool_h, pool_w, stride=1):
    """

    :param x: 2-dim ndarray
    :param pool_h: pooling window height
    :param pool_w: pooling window width
    :param stride: 보폭
    :return: max-pooling
    """
    h, w = x.shape[0], x.shape[1]  # 원본 데이터의 height/widtgh
    oh = (h - pool_h) // stride + 1  # 출력 배열의 height
    ow = (w - pool_w) // stride + 1  # 출력 배열의 width
    output = np.zeros((oh, ow))  # 출력 배열 초기화
    for i in range(oh):
        for j in range(ow):
            x_sub = x[(i * stride):(i * stride) + pool_h,
                    (j * stride):(j * stride) + pool_w]
            output[i, j] = np.max(x_sub)
    return output


if __name__ == '__main__':
    np.random.seed(114)
    x = np.random.randint(10, size=10)
    print(x)

    pooled = pooling1d(x, pool_size=2, stride=2)
    print(pooled)

    pooled = pooling1d(x, pool_size=4, stride=2)
    print(pooled)

    pooled = pooling1d(x, pool_size=4, stride=3)
    print(pooled)

    pooled = pooling1d(x, pool_size=3, stride=3)
    print(pooled)

    x = np.random.randint(100, size=(8, 8))
    print(x)

    pooled = pooling2d(x, pool_h=4, pool_w=4, stride=4)
    print(pooled)

    print()
    x = np.random.randint(100, size=(5, 5))
    print(x)
    pooled = pooling2d(x, pool_h=3, pool_w=3, stride=2)
    print(pooled)

    # MNIST 데이터 세트를 로드
    # 손글씨 이미지를 하나를 선택: shape=(1, 28, 28) -> (28, 28) 변환
    # 선택된 이미지를 pyplot을 사용해서 출력
    # window shape=(4, 4), stride=4 pooling -> output shape=(7,7)
    # pyplot으로 출력

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

    transformed_pic = pooling2d(num_img, 4, 4, 4)
    plt.imshow(transformed_pic, cmap='gray')
    plt.show()

    img = Image.open('desert.jpg')
    img_pixel = np.array(img)
    print(img_pixel.shape)
    img_r = img_pixel[:,:,0]
    img_g = img_pixel[:,:,1]
    img_b = img_pixel[:,:,2]

    tf_img_r=pooling2d(img_r, 32, 32, 32)
    tf_img_g=pooling2d(img_g, 32, 32, 32)
    tf_img_b=pooling2d(img_b, 32, 32, 32)
    plt.imshow(tf_img_r, cmap='pink_r')
    plt.show()
    plt.imshow(tf_img_g, cmap='Greens')
    plt.show()
    plt.imshow(tf_img_b, cmap='Blues')
    plt.show()
    tf_img_integrated=np.array([tf_img_r, tf_img_g, tf_img_b]).astype(np.uint8)
    print(tf_img_integrated.shape)
    tf_img_integrated = np.moveaxis(tf_img_integrated, 0, 2)
    print(tf_img_integrated.shape)
    plt.imshow(tf_img_integrated)
    plt.show()









