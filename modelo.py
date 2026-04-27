# Arquivo de utilidades para salvar e carregar o modelo treinado (pesos e RMSE)

import json
import os

def salvar_modelo(w, rmse, caminho):
    modelo = {
        "pesos": w,
        "rmse": rmse
    }
    with open(caminho, "w") as f:
        json.dump(modelo, f, indent=4)

def carregar_modelo(caminho):
    with open(caminho, "r") as f:
        modelo = json.load(f)
    return modelo["pesos"], modelo["rmse"]

def modelo_existe(caminho):
    return os.path.exists(caminho)