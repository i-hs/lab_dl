import numpy as np

a = np.arange(10)
print(a)
size = 7

for i in range(0, len(a), size):
    print(a[i:i+size])

print(a[10:11])

# 파이썬의 리스트에서의 append
b = []
c = [1, 2, 3]
b.append(c)
print(b)
b = []
c = [1, 2, 3]
b.extend(c)
print(b)

d = np.array([])
e = np.array([1, 2, 3])
f = np.append(d, e)
print('f:', f)

# numpy의 리스트에서의 append
x = np.array([1, 2])
y = np.array([3, 4, 5])
z = np.append(x, y)
print('z:',z)
