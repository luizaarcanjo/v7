from fastapi import FastAPI, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Funções para carregar os dados
def carregar_dados(caminho_campanhas='campanhas.csv', caminho_clientes='clientes.csv', caminho_transacoes='transacoes.csv'):
    """Carrega os dados dos arquivos CSV."""
    try:
        df_campanhas = pd.read_csv(caminho_campanhas)
        df_clientes = pd.read_csv(caminho_clientes)
        df_transacoes = pd.read_csv(caminho_transacoes)
        return df_campanhas, df_clientes, df_transacoes
    except FileNotFoundError as e:
        raise Exception(f"Erro ao carregar os dados: {e}. Certifique-se de que os arquivos CSV estão presentes.")

# Funções para Análise de Cluster
def realizar_clusterizacao(df_clientes, n_clusters=3):
    """Realiza a clusterização dos clientes e retorna o DataFrame com os rótulos."""
    X_cluster = df_clientes[["frequencia_compras", "total_gasto", "ultima_compra"]]
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clientes['cluster_numero'] = kmeans.fit_predict(X_cluster_scaled)

    cluster_means = df_clientes.groupby('cluster_numero')[["frequencia_compras", "total_gasto", "ultima_compra"]].mean()

    def mapear_cluster(cluster_numero):
        # Adapte esta lógica com base na análise das médias dos seus clusters
        if cluster_numero == 0:
            return "Cliente Ocasional"
        elif cluster_numero == 1:
            return "Cliente Frequente"
        elif cluster_numero == 2:
            return "Cliente de Alto Gasto"
        else:
            return str(cluster_numero)

    df_clientes['cluster_nome'] = df_clientes['cluster_numero'].apply(mapear_cluster)
    return df_clientes

def gerar_grafico_cluster(df_clientes):
    """Gera o gráfico de cluster e retorna a imagem como base64."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='frequencia_compras', y='total_gasto', hue='cluster_nome', data=df_clientes, palette='viridis')
    plt.title("Segmentação de Clientes com Base em Hábitos de Compra")
    plt.xlabel("Frequência de Compras")
    plt.ylabel("Total Gasto")
    plt.legend(title='Segmento de Cliente')

    # Salva o gráfico em um buffer na memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Funções para Análise Conjoint
def preparar_dados_conjoint(df_transacoes, df_campanhas):
    """Prepara os dados para a análise conjoint."""
    df_transacoes['campanha'] = pd.to_numeric(df_transacoes['campanha'], errors='coerce')
    df_conjoint_merge = pd.merge(df_transacoes, df_campanhas, left_on="campanha", right_on="campanha_id", how="left")
    return df_conjoint_merge

def gerar_dados_conjoint_simulados(n_respostas):
    """Gera dados conjoint simulados."""
    conjoint_data = {
        "desconto": np.random.choice([0, 1], size=n_respostas),
        "frete_gratis": np.random.choice([0, 1], size=n_respostas),
        "brindes": np.random.choice([0, 1], size=n_respostas),
        "escolha": np.random.choice([0, 1], size=n_respostas, p=[0.4, 0.6])
    }
    return pd.DataFrame(conjoint_data)

def realizar_analise_conjoint(df_conjoint_merge, df_conjoint):
    """Realiza a análise conjoint e retorna os resultados."""
    df_conjoint_merge = pd.concat([df_conjoint_merge, df_conjoint], axis=1)
    X = df_conjoint_merge[["desconto", "frete_gratis", "brindes"]]
    X = sm.add_constant(X)
    y = df_conjoint_merge["escolha"]

    model = sm.Logit(y, X)
    result = model.fit()

    return result

def gerar_grafico_importancia(result):
    """Gera o gráfico de importância dos atributos e o retorna como base64."""
    coeficientes = result.params.drop("const")
    plt.figure(figsize=(8, 6))
    coeficientes.plot(kind="bar", color="purple")
    plt.title("Importância dos Atributos nas Escolhas por Campanhas")
    plt.ylabel("Coeficientes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Salva o gráfico em um buffer na memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Funções para Regressão Linear
def realizar_regressao_linear(df_clientes):
    """Realiza a regressão linear para prever o total gasto."""

    X_reg = df_clientes[["idade", "renda_mensal", "frequencia_compras"]]
    y_reg = df_clientes[["total_gasto"]]

    model = LinearRegression()
    model.fit(X_reg, y_reg)

    # Adiciona a previsão ao DataFrame (cópia para não alterar o original)
    df_clientes_com_previsao = df_clientes.copy()
    df_clientes_com_previsao["total_gasto_previsto"] = model.predict(X_reg)

    coeficientes = model.coef_
    intercepto = model.intercept_  # Adicionando o intercepto
    return df_clientes_com_previsao, coeficientes, intercepto

def gerar_grafico_regressao(df_clientes_com_previsao):
    """Gera o gráfico de dispersão dos gastos reais vs. previstos e retorna como base64."""

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="total_gasto", y="total_gasto_previsto", data=df_clientes_com_previsao, color="skyblue")
    plt.title("Previsão de Gastos com Base nas Campanhas")
    plt.xlabel("Total Gasto Real")
    plt.ylabel("Total Gasto Previsto")
    plt.tight_layout()

    # Salva o gráfico em um buffer na memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Funções para CLV
def calcular_clv(df_clientes):
    """Calcula o CLV e identifica clientes de alto valor."""

    # Garante que estamos trabalhando com uma cópia para não modificar o original
    df_clientes_clv = df_clientes.copy()

    df_clientes_clv["clv"] = df_clientes_clv["total_gasto"] * df_clientes_clv["frequencia_compras"]
    df_clientes_clv["clv_valor_alto"] = df_clientes_clv["clv"] > df_clientes_clv["clv"].quantile(0.9)

    clientes_alto_valor = df_clientes_clv[df_clientes_clv["clv_valor_alto"]]
    clientes_alto_valor = clientes_alto_valor[["cliente_id", "clv"]] # Seleciona apenas as colunas relevantes

    return df_clientes_clv, clientes_alto_valor

def gerar_histograma_clv(df_clientes_clv):
    """Gera o histograma da distribuição do CLV e o retorna como base64."""

    plt.figure(figsize=(10, 6))  # Tamanho da figura para melhor visualização
    sns.histplot(df_clientes_clv["clv"], bins=25, kde=True, color="skyblue")
    plt.title("Distribuição do CLV")
    plt.xlabel("CLV")
    plt.ylabel("Frequência")
    plt.tight_layout()

    # Salva o gráfico em um buffer na memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Funções para CLV por Campanha
def juntar_dados(df_transacoes, df_clientes, df_campanhas):
    """Junta os DataFrames e calcula o CLV."""

    # Juntando transações com clientes
    df_trans_clientes = df_transacoes.merge(df_clientes, on="cliente_id", how="left")

    # Juntando com campanhas (usando o nome da campanha)
    df_completo = df_trans_clientes.merge(df_campanhas, left_on="campanha", right_on="nome_campanha", how="left")

    df_completo["clv"] = df_completo["total_gasto"] * df_completo["frequencia_compras"]

    return df_completo

def calcular_clv_por_campanha(df_completo):
    """Calcula o CLV médio por campanha."""

    clv_por_campanha = df_completo.groupby("campanha")["clv"].mean().sort_values(ascending=False)
    return clv_por_campanha

def gerar_grafico_clv_por_campanha(clv_por_campanha):
    """Gera o gráfico de CLV médio por campanha e o retorna como base64."""

    plt.figure(figsize=(10, 6))
    clv_por_campanha.plot(kind="bar", color="green")
    plt.title("CLV Médio por Campanha")
    plt.ylabel("CLV Médio")
    plt.xlabel("Campanha")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salva o gráfico em um buffer na memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Rotas da API
@app.post("/api/clusterizar")
async def clusterizar(n_clusters: int = 3):
    """API endpoint para realizar a clusterização."""
    try:
        df_campanhas, df_clientes, df_transacoes = carregar_dados()
        df_clientes_com_cluster = realizar_clusterizacao(df_clientes.copy(), n_clusters=n_clusters)
        grafico_base64 = gerar_grafico_cluster(df_clientes_com_cluster)

        # Retorna os dados do cluster e o gráfico
        return JSONResponse({
            'dados_cluster': df_clientes_com_cluster.to_dict(orient='records'),
            'grafico': grafico_base64
        })
    except Exception as e:
        return JSONResponse({'erro': str(e)}, 500)

@app.post("/api/conjoint")
async def conjoint():
    """API endpoint para realizar a análise conjoint."""
    try:
        df_campanhas, df_transacoes = carregar_dados()
        df_conjoint_merge = preparar_dados_conjoint(df_transacoes, df_campanhas)
        n_respostas = len(df_conjoint_merge)
        df_conjoint = gerar_dados_conjoint_simulados(n_respostas)
        result = realizar_analise_conjoint(df_conjoint_merge, df_conjoint)
        grafico_base64 = gerar_grafico_importancia(result)

        return JSONResponse({
            'resultados_conjoint': {"summary": str(result.summary())},
            'grafico': grafico_base64
        })
    except Exception as e:
        return JSONResponse({'erro': str(e)}, 500)

@app.post("/api/regressao")
async def regressao():
    """API endpoint para realizar a regressão linear."""
    try:
        df_campanhas, df_clientes, df_transacoes = carregar_dados()
        df_completo = juntar_dados(df_transacoes, df_clientes, df_campanhas)  # Assuming juntar_dados is defined
        df_clientes_reg = df_completo[["idade", "renda_mensal", "frequencia_compras", "total_gasto"]].drop_duplicates()
        df_clientes_com_previsao, coeficientes, intercepto = realizar_regressao_linear(df_clientes_reg)
        grafico_base64 = gerar_grafico_regressao(df_clientes_com_previsao)

        return JSONResponse({
            "coeficientes": coeficientes.tolist(),
            "intercepto": intercepto.tolist(),
            "dados_previsao": df_clientes_com_previsao.to_dict(orient="records"),
            "grafico": grafico_base64
        })
    except Exception as e:
        return JSONResponse({'erro': str(e)}, 500)

@app.post("/api/clv")
async def clv():
    """API endpoint para calcular o CLV."""
    try:
        df_campanhas, df_clientes, df_transacoes = carregar_dados()
        df_completo = juntar_dados(df_transacoes, df_clientes, df_campanhas)
        df_clientes_clv = df_completo[["cliente_id", "total_gasto", "frequencia_compras"]].drop_duplicates()
        df_clientes_com_clv, clientes_alto_valor = calcular_clv(df_clientes_clv)
        histograma_base64 = gerar_histograma_clv(df_clientes_com_clv)

        return JSONResponse({
            "dados_clv": df_clientes_com_clv.to_dict(orient="records"),
            "clientes_alto_valor": clientes_alto_valor.to_dict(orient="records"),
            "histograma": histograma_base64
        })
    except Exception as e:
        return JSONResponse({'erro': str(e)}, 500)

@app.post("/api/clv_campanha")
async def clv_campanha():
    """API endpoint para calcular o CLV por campanha."""
    try:
        df_campanhas, df_clientes, df_transacoes = carregar_dados()
        df_completo = juntar_dados(df_transacoes, df_clientes, df_campanhas)
        clv_por_campanha_data = calcular_clv_por_campanha(df_completo)
        grafico_base64 = gerar_grafico_clv_por_campanha(clv_por_campanha_data.reset_index())

        return JSONResponse({
            "clv_por_campanha": clv_por_campanha_data.to_dict(),
            "grafico": grafico_base64
        })
    except Exception as e:
        return JSONResponse({'erro': str(e)}, 500)