"""
MNIST 숫자 손글씨 데이터 세트
"""
from PIL import Image
import numpy as np

from dataset.mnist import load_mnist


def img_show(img_arr):
    """NumPy 배열(ndarray)로 작성된 이미지를 화면 출력"""

    img = Image.fromarray(np.uint8(img_arr))  # Numpy 배열 형식을 이미지로 변환
    img.show()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True,
                                                      flatten=False,
                                                      one_hot_label=True)
    # (학습 이미지 데이터, 학습 데이터 레이블), (테스트 이미지 데이터, 학습 데이터 레이블)

    print('X_train shape:', X_train.shape)
    # flatten=False인 경우, 이미지 구성을 (컬러, 가로, 세로) 형식으로 표시함.
    # (60000, 1, 28, 28): 28x28 크기의 흑백 이미지 60,000개
    print('y_train shape:', y_train.shape)
    # one_hot_label=True인 경우, one_hot_encoding 형식으로 숫자 레이블을 출력
    # (60000, 10)
    # 5 -> [0 0 0 0 0 5 0 0 0 0]
    print('y_train[0]:', y_train[0])

    # normalize=True인 경우, 각 픽셀의 숫자들이 0 ~ 1 사이의 숫자들로 정규화 된다.
    img = X_train[0]
    print(img)