import os

import leitura_arquivos
import modelo
import adaline

NOME_MODELO = "modelos/1.json"
ETA = 0.0025
MAX_EPOCAS = 1000

if __name__ == "__main__":
    opcao = input("Digite 1 para TREINAR ou 2 para USAR MODELO SALVO: ")

    if opcao == "1":
        treino = leitura_arquivos.carregar_dados("dados/treinamento.txt")        
        validacao = leitura_arquivos.carregar_dados_validacao("dados/validacao.txt")

        if modelo.modelo_existe(NOME_MODELO):
            w, rmse_antigo = modelo.carregar_modelo(NOME_MODELO)
            print(f"RMSE anterior: {rmse_antigo:.4f}")

            w = adaline.treinar(treino, w, ETA, MAX_EPOCAS)

        else:
            w = adaline.treinar(treino, None, ETA, MAX_EPOCAS)

        rmse = adaline.calcular_rmse(treino, w)
        print(f"RMSE atual: {rmse:.4f}")

        modelo.salvar_modelo(w, rmse, NOME_MODELO)
        
        saida = adaline.prever_classes(validacao, w)
        print(saida)

    elif opcao == "2":
        validacao = leitura_arquivos.carregar_dados_validacao("dados/validacao.txt")

        w, rmse = modelo.carregar_modelo(NOME_MODELO)

        print(f"RMSE do modelo carregado: {rmse:.4f}")

        saida = adaline.prever_classes(validacao, w)
        print(saida)

    else:
        print("Opção inválida")
