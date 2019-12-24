import numpy as np

def and_gate(x):
    # x는 [0, 0], [0, 1], [1, 0], [1, 1] 중 하나인 numpy.ndarray 타입
    # w = [w1, w2]인 numpy.ndarray 가중치와 bias b를 찾음
    result = np.logical_and(x[0], x[1])
    print(result)
    print(int(result))
    return int(result)

def nand_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test == 2:
        return 1
    else:
        return 0

def or_gate(x):
    result = np.logical_or(x[0], x[1])
    print(result)
    print(int(result))

    return int(result)


if __name__ == '__main__':
    y = np.array([1, 1])
    and_gate(y)
    nand_gate(y)
    or_gate(y)

