
import  numpy as np
class Class_:
    def __init__(self):
        self. W = np.array([1, 1])

    def trying(self, x):
        self.W += x
        print(self.W)

if __name__ == '__main__':
    abcd = Class_()
    abcd.trying(3)

