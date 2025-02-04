import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados processados
st.title("📊 Dashboard Financeiro - BanVic")

st.write("Bem-vindo ao painel de controle financeiro! Aqui você pode visualizar métricas importantes e acompanhar a evolução das transações ao longo do tempo.")

# Tentar carregar os dados
try:
    fato_transacoes = pd.read_csv("../data/fato_transacoes.csv")
    fato_transacoes["data_transacao"] = pd.to_datetime(fato_transacoes["data_transacao"])
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# KPIs principais
st.header("📌 Indicadores Financeiros")
receita_total = fato_transacoes["valor"].sum()
ticket_medio = fato_transacoes["valor"].mean()
num_clientes = fato_transacoes["id_cliente"].nunique()

st.metric(label="💰 Receita Total", value=f"R$ {receita_total:,.2f}")
st.metric(label="🛒 Ticket Médio", value=f"R$ {ticket_medio:,.2f}")
st.metric(label="👥 Número de Clientes", value=num_clientes)

# Distribuição das transações
st.header("📊 Análise de Transações")

st.subheader("📈 Distribuição dos Valores")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(fato_transacoes["valor"], bins=50, kde=True, ax=ax)
ax.set_title("Distribuição dos Valores das Transações")
st.pyplot(fig)

# Evolução ao longo do tempo
st.subheader("📊 Evolução do Volume de Transações")
transacoes_mensais = fato_transacoes.resample("M", on="data_transacao").agg({"valor": "sum"}).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(transacoes_mensais["data_transacao"], transacoes_mensais["valor"], marker="o", linestyle="-")
ax.set_title("Evolução das Transações Mensais")
ax.set_xlabel("Mês")
ax.set_ylabel("Valor Total Transacionado")
ax.grid(True)
st.pyplot(fig)

st.success("Dashboard atualizado com sucesso!")
