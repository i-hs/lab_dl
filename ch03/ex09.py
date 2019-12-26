"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.
"""
from PIL import Image
import numpy as np


def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""
    img = Image.open(image_file, mode='r')  # open image file
    print(type(img))  # ImageFile
    pixels_ = np.array(img)  # transfer image file object to numpy.ndarray type
    print('pixels shape:', pixels_.shape)  # (height, width, color)
    # color: 8bit(Gray scale), 24bit(RGB), 32bit(RGBA: RGB+불투명도)
    return pixels_


def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel) # ndarray 타입의 데이터를 이미지로 변환
    print(type(img))
    img.show()  # 이미지 뷰어를 사용해서 이미지 보기
    img.save(image_file)


if __name__ == '__main__':
    # image_to_pixel(), pixel_to_image() 함수 테스트
    pixels_1 = image_to_pixel('pengsoo.jpg')
    pixels_2 = image_to_pixel('ryan.png')

    pixel_to_image(pixels_1, 'pengsoo2.jpg')
    pixel_to_image(pixels_2, 'ryan2.png')
