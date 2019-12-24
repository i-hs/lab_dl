"""
Perceptron(퍼셉트론): 다수의 신호를 입력 받아서, 하나의 신호를 출력
"""

def and_gate(x1, x2):
    w1, w2 = 1, 1  # 가중치
    bias = -1
    y = x1 * w1 + x2 * w2 + bias
    if y > 0:
        return 1
    else:
        return 0


def nand_gate(x1, x2):
    w1, w2 = 1, 1
    bias = 0
    y = x1 * w1 + x2 * w2 + bias
    if y == 2 :
        return 0
    else:
        return 1


def or_gate(x1, x2):
    w1, w2 = 1, 1
    bias = 0
    y = x1 * w1 + x2 * w2 + bias
    if y == 0:
        return 0
    else:
        return 1


def xor_gate(x1, x2):
    """XOR(Exclusive OR: 배타적 OR)
    선형 관계식(y = x1 * w1 + x2 * w2 + b) 하나만 이용해서는 만들 수 없음
    NAND, OR, AND를 조합해야 가능
    """
    z1 = nand_gate(x1, x2)
    z2 = or_gate(x1, x2)
    return and_gate(z1, z2)  # forward propagation(순방향 전파)


if __name__== '__main__':
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'AND({x1}, {x2}) -> {and_gate(x1, x2)}')
            print(f'NAND({x1}, {x2}) -> {nand_gate(x1, x2)}')
            print(f'OR({x1}, {x2}) -> {or_gate(x1, x2)}')
            print(f'XOR({x1}, {x2}) -> {xor_gate(x1, x2)}')
            print(' ')