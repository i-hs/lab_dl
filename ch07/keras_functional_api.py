"""
keras functional api
"""

import numpy as np
from tensorflow.keras import Sequential, Model, Input, layers

# X: 입력: X(N, 64) -> Dense(32) -> ReLU -> Dense(10) -> Softmax

seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

seq_model.summary()
print()
# Keras의 함수형 API 기능을 사용해서 신경망을 생성하는 방법
# Input 객체 생성 -> 필요한 레이어 객체 생성 & 인스턴스 호출
# -> Model 객체를 생성
input_tensor = Input(shape=(64,))  # 입력 텐서의 shape을 설정
# 첫번째 은닉층(hidden layer) 생성 & 인스턴스 호출을 사용해서 입력 데이터 전달
x = layers.Dense(32, activation='relu')(input_tensor)
print(type(x))

# 두번째 은닉층 생성 & 첫번째 은닉층의 출력을 입력으로 전달
x = layers.Dense(32, activation='relu')(x)

# 출력층 생성 & 두번째 은닉층의 출력을 입력으로 전달
output_tensor = layers.Dense(10, activation='softmax')(x)

# 신경망 모델 생성 -input/output 텐서를 Model 생성자의 파라미터에 전달
model = Model(input_tensor, output_tensor)

# 모델 요약 정보
model.summary()


# 모델 생성 후
# -> 모델 컴파일 -> 모델 학습 -> 모델 평가, 예측
model.compile(optimizer= 'adam',
              loss='categorical_crossentropy')

# 가상의 데이터 생성
x_train = np.random.random(size=(1000, 64))
y_train = np.random.randint(10, size=(1000, 10))
print(x_train)
print(y_train)
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)
print(score)