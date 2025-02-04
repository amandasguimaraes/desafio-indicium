import pandas as pd

# Função para carregar os datasets
def load_data(file_name):
    try:
        print(f"🔄 Carregando {file_name}...")
        df = pd.read_csv(f"../data/{file_name}")
        print(f"✅ {file_name} carregado com sucesso!")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar {file_name}: {e}")
        return None

# Carregar os datasets
clientes = load_data("clientes.csv")
transacoes = load_data("transacoes.csv")
colaboradores = load_data("colaboradores.csv")
propostas = load_data("propostas_credito.csv")
agencias = load_data("agencias.csv")
contas = load_data("contas.csv")

# Criar tabelas dimensão
if clientes is not None:
    dim_clientes = clientes.rename(columns={"id": "id_cliente"})
if colaboradores is not None:
    dim_colaboradores = colaboradores.rename(columns={"id": "id_colaborador"})
if agencias is not None:
    dim_agencias = agencias.rename(columns={"id": "id_agencia"})
if contas is not None:
    dim_contas = contas.rename(columns={"id": "id_conta"})

# Criar tabelas fato
if transacoes is not None and clientes is not None and contas is not None:
    fato_transacoes = transacoes.merge(dim_clientes, on="id_cliente", how="left").merge(dim_contas, on="id_conta", how="left")
if propostas is not None and clientes is not None and colaboradores is not None:
    fato_propostas = propostas.merge(dim_clientes, on="id_cliente", how="left").merge(dim_colaboradores, on="id_colaborador", how="left")

# Salvar as tabelas em CSV
if clientes is not None:
    dim_clientes.to_csv("../data/dim_clientes.csv", index=False)
if colaboradores is not None:
    dim_colaboradores.to_csv("../data/dim_colaboradores.csv", index=False)
if agencias is not None:
    dim_agencias.to_csv("../data/dim_agencias.csv", index=False)
if contas is not None:
    dim_contas.to_csv("../data/dim_contas.csv", index=False)
if transacoes is not None:
    fato_transacoes.to_csv("../data/fato_transacoes.csv", index=False)
if propostas is not None:
    fato_propostas.to_csv("../data/fato_propostas.csv", index=False)

print("Data Warehouse atualizado com sucesso!")
