import numpy as np


class MultiplyLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, fruit, n):
        self.x = fruit
        self.y = n
        return self.x * self.y

    def backward(self, delta_output):
        dx = delta_output * self.y
        dy = delta_output * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x + self.y

    def backward(self, dout):
        dx, dy = dout, dout
        return dx, dy

if __name__ == '__main__':
    # MultiplyLayer 객체 생성
    apple_layer = MultiplyLayer()

    apple = 100  # 사과 1개의 가격 100원
    n = 2  # 사과 개수: 2개
    apple_price = apple_layer.forward(apple, n)  # 순방향 전파(forward propagation)
    print('사과 2개 가격:', apple_price)

    # tax_layer 변수를 MultiplyLayer 객체로 생성
    # tax = 1.1 설정해서 사과 2개 구매할 때 총 금액을 계산
    tax_layer = MultiplyLayer()

    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('사과 2개 구매시 총 금액:', total_price)

    # backward
    delta = 1.0
    dapple_price, d_tax = tax_layer.backward(delta)
    d_apple, d_number = apple_layer.backward(dapple_price)
    # 사과 개수가 1 증가하면 전체 가격은 얼마나 증가?
    print('사과 개수 1 증가시 전체 가격 증가:', d_number)
    # 사과 가격이 1 증가하면 전체 가격은 얼마나 증가?
    print('사과 가격 1 증가시 전체 가격 증가:', d_apple)
    # tax가 1 증가하면 전체 가격은 얼마가 증가?
    print('tax 1 증가시 전체 가격 증가:', d_tax)

    # AddLayer 테스트
    add_layer = AddLayer()
    x = 100
    y = 200
    dout = 1.5
    f = add_layer.forward(x, y)
    print(f)
    dx, dy = add_layer.backward(dout)
    print(dx, dy)