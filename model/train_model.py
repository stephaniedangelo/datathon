import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os

# Função para carregar os dados
def load_data(filepath='data/base_final.csv'):
    df = pd.read_csv(filepath)
    return df

# Função para preprocessamento e split
def preprocess_data(df, target_column='contratado'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded = encoder.fit_transform(X)

    return train_test_split(X_encoded, y, test_size=0.2, random_state=42), encoder

# Função para treinar o modelo
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Função principal
def main():
    print("🔍 Carregando dados...")
    df = load_data()

    print("✅ Pré-processando e dividindo os dados...")
    (X_train, X_test, y_train, y_test), encoder = preprocess_data(df)

    print("🎯 Treinando modelo RandomForest...")
    model = train_model(X_train, y_train)

    print("📊 Avaliando modelo...")
    y_pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    print("💾 Salvando modelo e encoder...")
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/modelo.pkl")
    joblib.dump(encoder, "model/encoder.pkl")

    print("🚀 Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()