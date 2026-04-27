# Arquivo de utilidades para carregar os dados de treinamento (X, D) e validação (X)

def carregar_dados(caminho):
    dados = []
    with open(caminho, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                valores = list(map(float, linha.split()))
                x = valores[:-1]
                d = valores[-1]
                dados.append((x, d))
    return dados

def carregar_dados_validacao(caminho):
    dados = []
    with open(caminho, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                valores = list(map(float, linha.split()))
                dados.append(valores)
    return dados
