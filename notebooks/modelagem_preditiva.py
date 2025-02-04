import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Carregar os dados
print("🔄 Carregando os dados de transações...")
try:
    transacoes = pd.read_csv("../data/fato_transacoes.csv")
    transacoes["data_transacao"] = pd.to_datetime(transacoes["data_transacao"], errors="coerce")
except Exception as e:
    print(f"❌ Erro ao carregar os dados: {e}")
    exit()

# Agregar transações por mês
print("📊 Processando os dados para análise...")
transacoes_mensais = transacoes.resample("M", on="data_transacao").agg({"valor": "sum"}).reset_index()

# Criar conjunto de treino e teste (80% treino, 20% teste)
train_size = int(len(transacoes_mensais) * 0.8)
train, test = transacoes_mensais[:train_size], transacoes_mensais[train_size:]

# 📌 Modelo ARIMA
print("📈 Treinando modelo ARIMA...")
model_arima = ARIMA(train["valor"], order=(5, 1, 0))
model_fit_arima = model_arima.fit()
predictions_arima = model_fit_arima.forecast(steps=len(test))

# Avaliação do ARIMA
mae_arima = mean_absolute_error(test["valor"], predictions_arima)
print(f"📉 Erro Médio Absoluto (ARIMA): {mae_arima:.2f}")

# 📌 Modelo Random Forest
print("🌲 Treinando modelo Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(np.arange(len(train)).reshape(-1, 1), train["valor"])
predictions_rf = rf_model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))

# Avaliação do Random Forest
mae_rf = mean_absolute_error(test["valor"], predictions_rf)
print(f"🌲 Erro Médio Absoluto (Random Forest): {mae_rf:.2f}")

# 📊 Comparação Visual
title = "📉 Comparação de Modelos Preditivos"
print(f"🔎 Gerando gráfico: {title}")
plt.figure(figsize=(12, 6))
plt.plot(test["data_transacao"], test["valor"], label="Real", marker="o")
plt.plot(test["data_transacao"], predictions_arima, label="ARIMA", linestyle="dashed")
plt.plot(test["data_transacao"], predictions_rf, label="Random Forest", linestyle="dashed")
plt.axvline(x=test["data_transacao"].iloc[0], color="r", linestyle="--", label="Início da Previsão")
plt.xlabel("Data")
plt.ylabel("Volume Transacionado")
plt.title(title)
plt.legend()
plt.grid(True)
plt.savefig("../visualizations/comparacao_modelos.png")
plt.show()

# Salvar previsões para o dashboard
print("💾 Salvando previsões para visualização...")
predictions_df = pd.DataFrame({
    "data_transacao": test["data_transacao"],
    "valor_real": test["valor"],
    "previsao_arima": predictions_arima,
    "previsao_rf": predictions_rf
})
predictions_df.to_csv("../data/previsoes.csv", index=False)

print("Modelagem preditiva concluída com sucesso!")
