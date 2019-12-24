"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.
"""
import pickle

import numpy as np


def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""

    with open(image_file, "rb") as image:
        f = image.read()
        # print(f)
        b = bytearray(f)
        print(b)
        c = np.array(b)
        # print(c.shape)
        print(c)
    return c


def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    with open(f'{image_file}.jpg', mode='wb') as f:  # w: write, b: binary
        pickle.dump(pixel, f)  # 객체(obj)를 파일(f)에 저장 -> serialization


if __name__ == '__main__':
    image_ = 'pengsoo.jpg'
    img_array = image_to_pixel(image_)
    # pixel_to_image(img_array, 'pengsoo2')

