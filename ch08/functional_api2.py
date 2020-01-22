
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, concatenate

"""
GoogleNet: p.271 그림 8-11
ResNet: p.272 그림 8-12
"""
# 입력 텐서 생성
input_tensor = Input(shape=(784,))
x1 = Dense(64, activation='relu')(input_tensor)
x2 = Dense(64, activation='relu')(input_tensor)

# 은닉층 생성
concat = concatenate([x1, x2])
x3 = Dense(32, activation='relu')(concat)

# 출력층 생성
output_tensor = Dense(10, activation='softmax')(x3)  # 두 개의 output 텐서를 연결

# 모델 생성
model = Model(input_tensor, output_tensor)
model.summary()


print()

input_tensor = Input(shape=(784,))
sc = Dense(32, activation='relu')(input_tensor)
x = Dense(32, activation='relu')(sc)
x = Dense(32, activation='relu')(x)
add = Add()([sc, x])
output_tensor = Dense(10, activation='softmax')(add)

model = Model(input_tensor, output_tensor)
model.summary()