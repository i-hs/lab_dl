import numpy as np


class Relu:
    """
    Relu(Rect ified Linear Unit)
    """


    def __init__(self):
        # relu 함수의 input 값(x)가 0보다 큰지 작은지를 저장할 field
        self.mask = None

    def forward(self, x):
        self.mask = (x < 0)
        return np.maximum(0, x)

    def backward(self, dout):
        # print('masking 전:', dout)
        dout[self.mask] = 0  # x가 음수(True) 이면 0으로 치환, 양수(False)이면 그대로
        # print('masking 후:', dout)
        return dout



if __name__ == '__main__':
    relu_gate = Relu()

    # x = 1 일 때 relu의 리턴 값
    y = relu_gate.forward(3)
    print('y:', y)

    np.random.seed(103)
    x = np.random.randn(5)
    print(x)
    y = relu_gate.forward(x)
    print('y:', y)

    print('mask:', relu_gate.mask)

    # back propagation(역전파)
    delta = np.random.randn(5)
    ex = relu_gate.backward(delta)


