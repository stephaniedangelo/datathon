# 🧠 Projeto Datathon – Machine Learning Engineering

## 👋 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Datathon Pós-Tech – Engenharia de Machine Learning**, com o objetivo de aplicar Inteligência Artificial para melhorar o processo de recrutamento e seleção da empresa **Decision**, especializada em alocação de talentos no setor de TI.

---

## 🎯 Objetivo

Criar um modelo de **Machine Learning** capaz de prever o **match entre candidatos e vagas**, considerando não apenas requisitos técnicos, mas também informações como formação, idiomas, e área de atuação — automatizando parte do processo seletivo.

---

## 🏗️ Estrutura do Projeto

```
📁 data/                # Base de dados em JSON e CSV final
📁 model/               # Scripts de treino e modelos salvos (.pkl)
📁 api/                 # API FastAPI com endpoint /predict
📁 notebooks/           # EDA exploratória em Jupyter
📁 tests/               # Testes unitários (a serem adicionados)
📄 train_model.py       # Script de treino do modelo
📄 requirements.txt     # Dependências do projeto
📄 README.md            # Este arquivo
```

---

## 🧪 Base de Dados

Utilizada base fornecida pela empresa Decision, composta por:

- `applicants.json` — informações de candidatos
- `vagas.json` — detalhes das vagas abertas
- `prospects.json` — histórico de tentativas de match entre candidatos e vagas

---

## 📊 Etapas Realizadas

- ✅ Análise Exploratória e Limpeza de Dados
- ✅ Transformação de JSONs estruturados em DataFrames tabulares
- ✅ Geração da base final com rótulo (`contratado`)
- ✅ Treinamento de modelo com `RandomForestClassifier`
- ✅ Criação e salvamento de encoder (`OrdinalEncoder`)
- ✅ Implementação de uma API com `FastAPI`
- ✅ Testes locais com Swagger UI (`/docs`)

---

## 🤖 Modelo de Machine Learning

- Algoritmo: `RandomForestClassifier`
- Features consideradas:
  - `nivel_academico`, `nivel_ingles`, `area_atuacao`, `tipo_contratacao`, `titulo_vaga`, etc.
- Target: `contratado` (0 = não contratado, 1 = contratado)
- Tratamento de valores desconhecidos com:  
  `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`

---

## 🚀 Como executar localmente

1. Clone o repositório:

```bash
git clone https://github.com/seuusuario/seurepositorio.git
cd seurepositorio
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Treine o modelo:

```bash
python model/train_model.py
```

4. Inicie a API:

```bash
uvicorn api.main:app --reload
```

4.1 Iniciar com Docker:  
  4.1.1 Builde da imagem:
    ```bash
    docker build -t datathon .
    ```
  4.1.2 Execucao da imagem:  
    ```bash
    docker build -it -p 8000:8000 datathon
    ```

5. Acesse a documentação da API:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📦 Exemplo de JSON para `/predict`

```json
{
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
```

---

## 📈 Resultados e Aprendizados

- O modelo obteve 100% de acurácia no conjunto de teste com os dados disponíveis, indicando que o rótulo `contratado` possui padrão claro na base fornecida.
- O maior desafio foi garantir que a API suportasse dados não vistos, resolvido com a configuração `handle_unknown='use_encoded_value'`.
- A API está pronta para receber novos dados e pode ser conectada a plataformas de recrutamento automatizado.

---
