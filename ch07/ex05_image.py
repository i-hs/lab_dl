"""
CNN(Convolutional Neural Network, 합성곱 신경망)
원래 convolution 연산은 영상/음성 처리(image/audio processing)에서 신호를 변환하기 위한 연산
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate

# jpg 파일 오픈
img = Image.open('desert.jpg')
img_pixel = np.array(img)
print(img_pixel.shape)
# 머신 러닝 라이브러리에 따라서 color 표기의 위치가 달라진다.


# plt.imshow(img_pixel)  # pixel 변환된 ndarray를 전달
# plt.show()
# 이미지의 RED 값 정보
img_red = img_pixel[:, :, 0]
print(img_red)
# (3, 3, 3) 필터
filter = np.zeros((3, 3, 3))
print(filter)
filter[1, 1, 0] = 2.0
print(filter)
transformed = convolve(img_pixel, filter, mode='same')
plt.imshow((transformed * 255).astype(np.uint8))
plt.show()