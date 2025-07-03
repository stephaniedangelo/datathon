from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Carregar modelo e encoder
model = joblib.load("model/modelo_recrutamento.pkl")
encoder = joblib.load("model/encoder.pkl")

# Definir os campos que a API espera receber
class MatchInput(BaseModel):
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

@app.get("/")
def home():
    return {"status": "API está no ar 🚀"}

@app.post("/predict")
def predict_match(data: MatchInput):
    input_data = [[
        data.nivel_academico_x,
        data.nivel_ingles_x,
        data.nivel_espanhol_x,
        data.area_atuacao,
        data.titulo_profissional,
        data.nivel_ingles_y,
        data.nivel_espanhol_y,
        data.areas_atuacao,
        data.titulo_vaga,
        data.tipo_contratacao
    ]]
    
    try:
        input_encoded = encoder.transform(input_data)
        prediction = model.predict(input_encoded)
        return {"match": bool(prediction[0])}
    except Exception as e:
        return {"erro": str(e)}