import joblib
import pandas as pd
import numpy as np
import pytest

def test_model_and_encoder_load():
    model = joblib.load("model/modelo.pkl")
    encoder = joblib.load("model/encoder.pkl")
    assert model is not None
    assert encoder is not None

def test_model_prediction_shape():
    model = joblib.load("model/modelo.pkl")
    encoder = joblib.load("model/encoder.pkl")

    sample_data = pd.DataFrame([{
        "nivel_academico_x": "Ensino Superior Completo",
        "nivel_ingles_x": "Avançado",
        "nivel_espanhol_x": "Nenhum",
        "area_atuacao": "TI - Desenvolvimento/Programação-",
        "titulo_profissional": "Analista Desenvolvedor",
        "nivel_ingles_y": "Avançado",
        "nivel_espanhol_y": "Nenhum",
        "areas_atuacao": "TI - Desenvolvimento/Programação-",
        "titulo_vaga": "Analista Desenvolvedor Java",
        "tipo_contratacao": "CLT Full"
    }])

    encoded = encoder.transform(sample_data)
    pred = model.predict(encoded)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (1,)