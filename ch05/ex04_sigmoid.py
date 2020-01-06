import numpy as np

def sigmoid(x):
    """ sigmoid = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, delta_output):
        return delta_output * self.y * (1 - self.y)

if __name__ == '__main__':
    sigmoiding = Sigmoid()
    sig_fw = sigmoiding.forward(0)
    print(sig_fw)
    sig_bw = sigmoiding.backward(1)
    print(sig_bw)