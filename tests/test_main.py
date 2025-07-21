import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_match():
    sample_input = {
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
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "match" in response.json()
    assert isinstance(response.json()["match"], bool)