import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W  # weight 행렬
        self.b = b  # bias 행렬
        self.X = None  # 입력 행렬을 저장할 field
        self.dW = None  # W 행렬 gradient  -> W = W - lr * dW
        self.db = None  # b 행렬 gradient  -> b = b - lr * db

    def forward(self, X):
        self.X = X  # 역전파에서 사용됨
        out = self.X.dot(self.W) + self.b
        return out

    def backward(self, dout):
        # b 행렬 방향으로의 gradient
        self.db = np.sum(dout, axis=0)
        # Z 행렬 방향으로의 gradient -> W 방향, X 방향으로의 gradient
        self.dW = self.X.T.dot(dout)  # GD를 이용해서 W, b를 fitting 시킬 때 사용하기 위해서
        dX = dout.dot(self.W.T)
        return dX







if __name__ == '__main__':
    np.random.seed(103)
    X = np.random.randint(10, size=(2, 3))  # 입력 행렬
    print('X:', X)
    W = np.random.randint(10, size=(3, 5))  # 가중치 행렬
    print('W:', W)
    b = np.random.randint(10, size=5)  # bias 행렬
    print('b:', b)
    affine = Affine(W, b)  # Affine 클래스의 객체 생성
    Y = affine.forward(X)  # Affine의 출력 값
    print('Y =', Y)

    dout = np.random.randn(2, 5)
    dX = affine.backward(dout)
    print('dX=', dX)
    print('dW=', affine.dW)
    print('db=', affine.db)
