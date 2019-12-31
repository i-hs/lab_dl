"""
numpy.nditer 객체: 반복문(for, while)을 쓰기 쉽게 도와주는 객체
"""
import numpy as np

np.random.seed(1231)
a = np.random.randint(100, size=(2, 3))
print(a)

# 40 21 5 52 84 39
for row in a:
    for x in row:
        print(x, end=' ')
print()

i = 0
while i < a.shape[0]:
    j = 0
    while j < a.shape[1]:
        print(a[i, j], end=' ')
        j += 1
    i += 1
print()

with np.nditer(a) as iterator:  # nditer 클래스 객체 생성
    for val in iterator:
        print(val, end=' ')
print()

with np.nditer(a, flags=['multi_index']) as iterator:
    while not iterator.finished:
        i = iterator.multi_index
        print(f'{i}:{a[i]}', end=' ')
        iterator.iternext()
print()

with np.nditer(a, flags=['c_index']) as iterator:
    while not iterator.finished:
        i = iterator.index
        print(f'[{i}]{iterator[0]}', end=' ')
        iterator.iternext()
print()

a = np.arange(6).reshape((2, 3))
print(a)
with np.nditer(a, flags=['multi_index']) as it:
    while not it.finished:
        a[it.multi_index] *= 2
        it.iternext()
print(a)


a = np.arange(6).reshape((2, 3))
with np.nditer(a, flags=['c_index'], op_flags=['readwrite']) as it:
    while not it.finished:
        it[0] *= 2
        it.iternext()
print(a)




