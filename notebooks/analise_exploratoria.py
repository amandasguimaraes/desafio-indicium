import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Carregar datasets

def load_data(file_name):
    try:
        df = pd.read_csv(f"../data/{file_name}")
        df = df.replace(0, np.nan)  # Melhor manter NaN para não distorcer os dados
        print(f"Carregamos {file_name} com sucesso!")
        return df
    except Exception as e:
        print(f"Erro ao abrir {file_name}: {e}")
        return None

datasets = {
    "Clientes": load_data("clientes.csv"),
    "Transacoes": load_data("transacoes.csv"),
    "Colaboradores": load_data("colaboradores.csv"),
    "Propostas": load_data("propostas_credito.csv"),
    "Agencias": load_data("agencias.csv"),
    "Contas": load_data("contas.csv")
}

datasets = {k: v for k, v in datasets.items() if v is not None}  # Remove arquivos não carregados

# Ajustar formato de data
datasets["Transacoes"]["data_transacao"] = pd.to_datetime(datasets["Transacoes"]["data_transacao"], errors="coerce")

def tratar_dados(df, colunas_obrigatorias):
    for col in colunas_obrigatorias:
        if col not in df.columns:
            print(f"Aviso: Coluna {col} não encontrada.")
    df.fillna(method='ffill', inplace=True)  # Preenchimento progressivo
    return df

datasets["Transacoes"] = tratar_dados(datasets["Transacoes"], ["id_cliente", "valor", "data_transacao"])

# KPIs
receita_total = datasets["Transacoes"]["valor"].sum()
ticket_medio = datasets["Transacoes"]["valor"].mean()
transacoes_por_cliente = datasets["Transacoes"].groupby("id_cliente")["valor"].count().mean()

print(f"📊 Receita Total: R${receita_total:,.2f}")
print(f"💳 Ticket Médio: R${ticket_medio:,.2f}")
print(f"👥 Média de Transações por Cliente: {transacoes_por_cliente:.2f}")

# Modelagem preditiva
transacoes_mensais = datasets["Transacoes"].resample("M", on="data_transacao").agg({"valor": "sum"}).reset_index()

train_size = int(len(transacoes_mensais) * 0.8)
train, test = transacoes_mensais[:train_size], transacoes_mensais[train_size:]

# ARIMA
print("Treinando modelo ARIMA...")
model = ARIMA(train["valor"], order=(3, 1, 2))
model_fit = model.fit()
predictions_arima = model_fit.forecast(steps=len(test))
mae_arima = mean_absolute_error(test["valor"], predictions_arima)
print(f"Erro médio (ARIMA): {mae_arima:.2f}")

# Random Forest
print("Treinando modelo Random Forest...")
X_train = np.arange(len(train)).reshape(-1, 1)
X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, train["valor"])
predictions_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(test["valor"], predictions_rf)
print(f"Erro médio (Random Forest): {mae_rf:.2f}")

# LSTM
print("Treinando modelo LSTM...")
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train["valor"].values.reshape(-1, 1))
test_scaled = scaler.transform(test["valor"].values.reshape(-1, 1))

X_train, y_train = [], []
for i in range(1, len(train_scaled)):
    X_train.append(train_scaled[i-1:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model_lstm = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dense(units=1)
])
model_lstm.compile(optimizer="adam", loss="mean_squared_error")
model_lstm.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

X_test = np.array([test_scaled[i-1:i, 0] for i in range(1, len(test_scaled))])
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)
mae_lstm = mean_absolute_error(test["valor"].iloc[1:], predictions_lstm)
print(f"Erro médio (LSTM): {mae_lstm:.2f}")

print("Análise finalizada!")
