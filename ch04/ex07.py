"""
경사 하강법(gradient descent)
x_new = x - lr * df/dx
위 과정을 반복 -> f(x)의 최솟값을 찾음
"""
import numpy as np
import matplotlib.pyplot as plt


def partial_gradient_dim_1(fn, x):
    x = x.astype(np.float)  # 실수 타입
    gradient = np.zeros_like(x)  # np.zeros(shape=x.shape)
    h = 1e-4  # 0.0001
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        fh1 = fn(x)
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2 * h)
        x[i] = ith_value
    return gradient


def partial_gradient(fn, x):
    """x = [
        [x11, x12, x13 ..],
        [x21, x22, x23 ..],
        ..
    ]
    """
    if x.ndim == 1:
        return partial_gradient_dim_1(fn, x)

    else:
        gradient = np.zeros_like(x)
        for i, x_ in enumerate(x):
            print('x_i:', x_)
            gradient[i] = partial_gradient_dim_1(fn, x_)
        print('gradient:', gradient)
        return gradient


def gradient_method(fn, x_init, lr=0.1, step=100):
    x = x_init  # 점진적으로 변화시킬 변수
    x_history = []  # x가 변화되는 과정을 저장할 배열
    for i in range(step):  # step 회수만큼 반복하면서
        x_history.append(x.copy())  # x의 복사본을 x 변화 과정에 기록
        grad = partial_gradient(fn, x)  # x에서의 gradient를 계산
        x -= lr * grad  # x_new = x_init - lr * grad: x를 변경 lr : 변화율

    return x, np.array(x_history)


def fn(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


if __name__ == '__main__':
    init_x = np.array([4.0])
    x, x_hist = gradient_method(fn, init_x, lr=0.2, step=200)
    print('x =', x)
    print('x_hist =', x_hist)

    # 학습률(learning rate: lr) 이 너무 작으면(lr = 0.001),
    # 최소값을 찾아가는 시간이 너무 오래 걸린다.
    # 학습률이 너무 크면(lr = 1.5), 최솟값을 찾지 못하고 발산할 수 있다.

    init_x = np.array([4., -3.])
    x, x_hist = gradient_method(fn, init_x, lr=0.1, step=50)
    print('x =', x)
    print('x_hist =', x_hist)

    # x_hist(최소값을 찾아가는 과정)을 산점도 그래프로 표현
    plt.scatter(x_hist[:, 0], x_hist[:, 1])
    # 동심원
    for r in range(1, 5):
        r = float(r)  # 정수 -> 실수 변환
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r**2 - x_pts**2)  # 위쪽 반원
        y_pts2 = -np.sqrt(r**2 - x_pts**2)  # 아래쪽 반원
        plt.plot(x_pts, y_pts1, ':', color = 'gray')
        plt.plot(x_pts, y_pts2, ':', color = 'gray')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color='0.8')
    plt.axhline(color='0.8')
    plt.show()
