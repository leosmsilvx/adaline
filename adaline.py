import random
import math

def sinal(u):
    return 1 if u >= 0 else -1

def produto_escalar(w, x):
    return sum(wi * xi for wi, xi in zip(w[:-1], x)) + w[-1]

def calcular_rmse(dados, w):
    soma_erro = 0
    for x, d in dados:
        y = produto_escalar(w, x)
        soma_erro += (d - y) ** 2
    return math.sqrt(soma_erro / 2)

def treinar(dados, w, eta, max_epocas, tolerancia=1e-12, alvo_rmse=0.01):
    n = len(dados[0][0])
    if w is None:
        w = [random.uniform(-1, 1) for _ in range(n + 1)]

    print(f"Pesos iniciais: {w}")

    rmse_anterior = float("inf")
    epoca = 0;
    while True:
        for x, d in dados:
            y = produto_escalar(w, x)
            erro = d - y
            for i in range(n):
                w[i] += eta * erro * x[i]
            w[-1] += eta * erro

        rmse_atual = calcular_rmse(dados, w)
        epoca += 1
        print(f"Época {epoca} - RMSE: {rmse_atual}")

        if abs(rmse_anterior - rmse_atual) < tolerancia:
            break

        rmse_anterior = rmse_atual

    print(f"Pesos finais: {w}")
    return w

def prever_classes(dados, w):
    return [sinal(produto_escalar(w, x)) for x in dados]
