from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Inicialização da API
app = FastAPI(title="API de Match Candidato-Vaga", version="1.0")

# Carregamento do modelo e encoder
try:
    model = joblib.load("model/modelo.pkl")
    encoder = joblib.load("model/encoder.pkl")
    logging.info("Modelo e encoder carregados com sucesso.")
except Exception as e:
    logging.error("Erro ao carregar modelo ou encoder: %s", str(e))
    raise

# Schema de entrada usando Pydantic
class Item(BaseModel):
    nivel_academico_x: str
    nivel_ingles_x: str
    nivel_espanhol_x: str
    area_atuacao: str
    titulo_profissional: str
    nivel_ingles_y: str
    nivel_espanhol_y: str
    areas_atuacao: str
    titulo_vaga: str
    tipo_contratacao: str

# Rota principal
@app.post("/predict")
def predict(data: Item):
    try:
        logging.info(f"Input recebido: {data}")
        df = pd.DataFrame([data.dict()])
        df_encoded = encoder.transform(df)
        pred = model.predict(df_encoded)
        proba = model.predict_proba(df_encoded)[0][1] if hasattr(model, "predict_proba") else None

        response = {
            "match": bool(pred[0]),
            "mensagem": "Candidato compatível" if pred[0] else "Candidato não compatível"
        }

        if proba is not None:
            response["probabilidade_match"] = round(float(proba), 3)

        return response
    except Exception as e:
        logging.error("Erro na predição: %s", str(e))
        raise HTTPException(status_code=500, detail="Erro ao processar a previsão.")