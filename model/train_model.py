import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
import joblib

# Carrega o dataset pronto
df = pd.read_csv('data/dataset_final.csv')  # opcionalmente salve antes no notebook

# Seleciona colunas para o modelo
features = [
    'nivel_academico_x', 'nivel_ingles_x', 'nivel_espanhol_x',
    'area_atuacao', 'titulo_profissional',
    'nivel_ingles_y', 'nivel_espanhol_y',
    'areas_atuacao', 'titulo_vaga', 'tipo_contratacao'
]

df_model = df[features + ['contratado']].fillna('Desconhecido')

# Codifica os dados categóricos
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X = encoder.fit_transform(df_model[features])
y = df_model['contratado']

# Divide treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Salva o modelo e o encoder
joblib.dump(model, 'model/modelo_recrutamento.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
