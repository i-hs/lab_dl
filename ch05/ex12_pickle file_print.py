import pickle
"""ex11.py에서 저장한 pickle 파일을 읽어서,
파라미터(가중치/편향 행렬)들을 화면에 출력"""

with open('neural_net_params.pickle', mode='rb') as f: # r: read, b: binary
    data = pickle.load(f)
print(data)