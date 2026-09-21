import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Laboratório Estatístico Interativo",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# NÚCLEO ESTATÍSTICO PRÓPRIO
# ============================================================

def validar_dados(dados, minimo=1):
    """Valida e transforma os dados em vetor numérico."""
    dados = np.asarray(dados, dtype=float)

    if dados.size < minimo:
        raise ValueError(
            f"São necessárias pelo menos {minimo} observações."
        )

    if np.isnan(dados).any():
        raise ValueError("Os dados possuem valores ausentes.")

    if np.isinf(dados).any():
        raise ValueError("Os dados possuem valores infinitos.")

    return dados


def calcular_media(dados):
    dados = validar_dados(dados)
    return np.sum(dados) / len(dados)


def calcular_mediana(dados):
    dados = validar_dados(dados)
    ordenados = np.sort(dados)

    n = len(ordenados)
    meio = n // 2

    if n % 2 == 0:
        return (ordenados[meio - 1] + ordenados[meio]) / 2

    return ordenados[meio]


def calcular_variancia(dados, ddof=1):
    dados = validar_dados(dados, minimo=2)

    n = len(dados)

    if n - ddof <= 0:
        raise ValueError("ddof inválido para o tamanho da amostra.")

    media = calcular_media(dados)

    return np.sum((dados - media) ** 2) / (n - ddof)


def calcular_desvio_padrao(dados, ddof=1):
    return np.sqrt(calcular_variancia(dados, ddof))


def calcular_amplitude(dados):
    dados = validar_dados(dados)
    return np.max(dados) - np.min(dados)


def calcular_quartis(dados):
    dados = validar_dados(dados)

    q1 = np.percentile(dados, 25)
    q2 = np.percentile(dados, 50)
    q3 = np.percentile(dados, 75)

    return q1, q2, q3


def calcular_iqr(dados):
    q1, _, q3 = calcular_quartis(dados)
    return q3 - q1


def calcular_limites_outliers(dados):
    q1, _, q3 = calcular_quartis(dados)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    return limite_inferior, limite_superior


def calcular_correlacao(x, y):
    x = validar_dados(x, minimo=2)
    y = validar_dados(y, minimo=2)

    if len(x) != len(y):
        raise ValueError("As variáveis precisam possuir o mesmo tamanho.")

    media_x = calcular_media(x)
    media_y = calcular_media(y)

    numerador = np.sum((x - media_x) * (y - media_y))

    denominador = np.sqrt(
        np.sum((x - media_x) ** 2)
        * np.sum((y - media_y) ** 2)
    )

    if denominador == 0:
        raise ValueError("Não é possível calcular a correlação.")

    return numerador / denominador


# ============================================================
# REGRESSÃO LINEAR POR MMQ
# ============================================================

def regressao_mmq(x, y):
    """
    Calcula regressão linear simples pelo Método dos Mínimos Quadrados.
    y = b0 + b1*x
    """
    x = validar_dados(x, minimo=3)
    y = validar_dados(y, minimo=3)

    if len(x) != len(y):
        raise ValueError("X e Y precisam ter o mesmo número de observações.")

    x_media = calcular_media(x)
    y_media = calcular_media(y)

    sxx = np.sum((x - x_media) ** 2)
    sxy = np.sum((x - x_media) * (y - y_media))

    if sxx == 0:
        raise ValueError("A variável X não possui variação suficiente.")

    b1 = sxy / sxx
    b0 = y_media - b1 * x_media

    y_estimado = b0 + b1 * x

    residuos = y - y_estimado

    sse = np.sum(residuos ** 2)
    sst = np.sum((y - y_media) ** 2)

    if sst == 0:
        r2 = 1.0
    else:
        r2 = 1 - (sse / sst)

    return b0, b1, y_estimado, residuos, r2


def intervalo_predicao(x, y, x_novo, confianca=0.95):
    """
    Calcula a predição pela regressão MMQ e um intervalo
    de predição para uma nova observação.
    """
    x = validar_dados(x, minimo=3)
    y = validar_dados(y, minimo=3)

    b0, b1, _, residuos, r2 = regressao_mmq(x, y)

    n = len(x)
    graus_liberdade = n - 2

    x_media = calcular_media(x)

    sxx = np.sum((x - x_media) ** 2)

    if sxx == 0:
        raise ValueError("Não há variação suficiente em X.")

    erro_padrao = np.sqrt(
        np.sum(residuos ** 2) / graus_liberdade
    )

    previsao = b0 + b1 * x_novo

    alfa = 1 - confianca

    t_critico = stats.t.ppf(
        1 - alfa / 2,
        graus_liberdade
    )

    erro_predicao = (
        t_critico
        * erro_padrao
        * np.sqrt(
            1
            + 1 / n
            + ((x_novo - x_media) ** 2) / sxx
        )
    )

    limite_inferior = previsao - erro_predicao
    limite_superior = previsao + erro_predicao

    return (
        previsao,
        limite_inferior,
        limite_superior,
        r2
    )


# ============================================================
# TESTES DO NÚCLEO
# ============================================================

def executar_testes():
    dados = np.array([1, 2, 3, 4, 5])

    assert calcular_media(dados) == 3
    assert calcular_mediana(dados) == 3

    assert calcular_variancia(
        dados,
        ddof=0
    ) == 2

    assert calcular_variancia(
        dados,
        ddof=1
    ) == 2.5

    assert np.isclose(
        calcular_desvio_padrao(dados, ddof=0),
        np.sqrt(2)
    )

    assert np.isclose(
        calcular_desvio_padrao(dados, ddof=1),
        np.sqrt(2.5)
    )

    assert calcular_amplitude(dados) == 4
    assert calcular_iqr(dados) == 2

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    assert np.isclose(
        calcular_correlacao(x, y),
        1
    )

    b0, b1, _, _, r2 = regressao_mmq(x, y)

    assert np.isclose(b0, 0)
    assert np.isclose(b1, 2)
    assert np.isclose(r2, 1)

    return True


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data
def carregar_dados():
    return pd.read_csv("dados/dataset.csv")


df = carregar_dados()


# ============================================================
# CABEÇALHO
# ============================================================

st.title("📊 Laboratório Estatístico Interativo")

st.write(
    "Aplicação para exploração, análise estatística, simulação, "
    "correlação e regressão utilizando o conjunto de dados "
    "Bike Sharing."
)

st.success(
    f"Dataset carregado com sucesso: "
    f"{len(df):,} registros e {len(df.columns)} variáveis."
)


# ============================================================
# VISÃO GERAL
# ============================================================

st.header("📋 Visão geral dos dados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Registros",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Variáveis",
        len(df.columns)
    )

with col3:
    st.metric(
        "Valores ausentes",
        int(df.isna().sum().sum())
    )

with col4:
    st.metric(
        "Variáveis numéricas",
        len(df.select_dtypes(include=np.number).columns)
    )


# ============================================================
# DADOS
# ============================================================

st.subheader("Dados")

st.dataframe(
    df,
    width="stretch",
    height=400
)


# ============================================================
# TIPOS DAS VARIÁVEIS
# ============================================================

st.subheader("Tipos das variáveis")

tipos = pd.DataFrame({
    "Variável": df.columns,
    "Tipo": df.dtypes.astype(str).values
})

st.dataframe(
    tipos,
    width="stretch"
)


# ============================================================
# CRITÉRIO 1
# NÚCLEO PRÓPRIO + VALIDAÇÃO
# ============================================================

st.header("🧮 1. Núcleo estatístico próprio e validação")

st.write(
    "As principais medidas estatísticas utilizadas na aplicação "
    "foram implementadas por meio de funções próprias."
)

try:
    executar_testes()

    st.success(
        "✅ Todos os testes do núcleo estatístico foram aprovados."
    )

except Exception as erro:
    st.error(
        f"❌ Falha nos testes: {erro}"
    )


st.subheader("Demonstração das funções próprias")

variaveis_nucleo = [
    "cnt",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

variavel_nucleo = st.selectbox(
    "Escolha uma variável:",
    variaveis_nucleo,
    key="nucleo_variavel"
)

dados_nucleo = df[variavel_nucleo].dropna().values

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Média",
        f"{calcular_media(dados_nucleo):.4f}"
    )

with col2:
    st.metric(
        "Mediana",
        f"{calcular_mediana(dados_nucleo):.4f}"
    )

with col3:
    st.metric(
        "Amplitude",
        f"{calcular_amplitude(dados_nucleo):.4f}"
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Variância (ddof=0)",
        f"{calcular_variancia(dados_nucleo, 0):.4f}"
    )

with col5:
    st.metric(
        "Variância (ddof=1)",
        f"{calcular_variancia(dados_nucleo, 1):.4f}"
    )

with col6:
    st.metric(
        "Desvio-padrão (ddof=1)",
        f"{calcular_desvio_padrao(dados_nucleo, 1):.4f}"
    )

st.info(
    "ddof=0 representa a variância populacional, enquanto "
    "ddof=1 representa a variância amostral."
)


# ============================================================
# CRITÉRIO 2
# DESCRITIVA INTERATIVA
# ============================================================

st.header("📊 2. Estatística descritiva interativa")

variavel_descritiva = st.selectbox(
    "Escolha uma variável numérica:",
    df.select_dtypes(include=np.number).columns,
    key="descritiva_variavel"
)

dados = df[variavel_descritiva].dropna().values

q1, mediana, q3 = calcular_quartis(dados)
iqr = calcular_iqr(dados)

limite_inferior, limite_superior = calcular_limites_outliers(
    dados
)

st.subheader("Medidas descritivas")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Média",
        f"{calcular_media(dados):.4f}"
    )

with col2:
    st.metric(
        "Mediana",
        f"{mediana:.4f}"
    )

with col3:
    st.metric(
        "Desvio-padrão",
        f"{calcular_desvio_padrao(dados, 1):.4f}"
    )

with col4:
    st.metric(
        "IQR",
        f"{iqr:.4f}"
    )


# ------------------------------------------------------------
# STURGES
# ------------------------------------------------------------

n = len(dados)

numero_classes = int(
    np.ceil(1 + np.log2(n))
)

st.subheader("Distribuição de frequências")

st.write(
    f"O número de classes foi definido pela regra de Sturges: "
    f"**{numero_classes} classes** para {n:,} observações."
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    dados,
    bins=numero_classes,
    edgecolor="black"
)

ax.set_title(
    f"Distribuição de {variavel_descritiva}"
)

ax.set_xlabel(variavel_descritiva)
ax.set_ylabel("Frequência")

st.pyplot(fig)


# ------------------------------------------------------------
# IQR E OUTLIERS
# ------------------------------------------------------------

st.subheader("Análise do intervalo interquartil")

st.write(
    f"Q1 = **{q1:.4f}** | "
    f"Mediana = **{mediana:.4f}** | "
    f"Q3 = **{q3:.4f}**"
)

st.write(
    f"Limite inferior para possíveis outliers: "
    f"**{limite_inferior:.4f}**"
)

st.write(
    f"Limite superior para possíveis outliers: "
    f"**{limite_superior:.4f}**"
)

outliers = dados[
    (dados < limite_inferior)
    | (dados > limite_superior)
]

st.metric(
    "Possíveis outliers",
    f"{len(outliers):,}"
)

if len(outliers) == 0:
    st.success(
        "Interpretação: não foram identificados possíveis "
        "outliers pelo critério de 1,5 × IQR."
    )
else:
    percentual_outliers = (
        len(outliers) / len(dados)
    ) * 100

    st.info(
        f"Interpretação: foram identificadas {len(outliers):,} "
        f"observações além dos limites do IQR, representando "
        f"{percentual_outliers:.2f}% dos dados."
    )


# ============================================================
# CRITÉRIO 3
# SIMULAÇÃO E DISTRIBUIÇÕES
# ============================================================

st.header("🎲 3. Simulação e distribuições")

st.write(
    "As simulações utilizam a própria variável selecionada no "
    "conjunto de dados para ilustrar a Lei dos Grandes Números "
    "e o Teorema Central do Limite."
)

variavel_simulacao = st.selectbox(
    "Variável utilizada nas simulações:",
    [
        "cnt",
        "temp",
        "atemp",
        "hum",
        "windspeed"
    ],
    key="simulacao_variavel"
)

dados_simulacao = df[
    variavel_simulacao
].dropna().values


# ------------------------------------------------------------
# LGN
# ------------------------------------------------------------

st.subheader("Lei dos Grandes Números — LGN")

tamanho_amostra_lgn = st.slider(
    "Número máximo de observações acumuladas:",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    key="lgn_tamanho"
)

rng_lgn = np.random.default_rng(42)

amostra_lgn = rng_lgn.choice(
    dados_simulacao,
    size=tamanho_amostra_lgn,
    replace=True
)

media_acumulada = np.cumsum(
    amostra_lgn
) / np.arange(
    1,
    tamanho_amostra_lgn + 1
)

media_populacional = calcular_media(
    dados_simulacao
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    media_acumulada,
    label="Média acumulada"
)

ax.axhline(
    media_populacional,
    linestyle="--",
    label="Média dos dados"
)

ax.set_title(
    "Convergência da média amostral"
)

ax.set_xlabel("Número de observações")
ax.set_ylabel("Média")

ax.legend()

st.pyplot(fig)

st.info(
    f"A média dos dados é aproximadamente "
    f"**{media_populacional:.4f}**. "
    "À medida que o tamanho da amostra aumenta, "
    "a média acumulada tende a se aproximar desse valor."
)


# ------------------------------------------------------------
# TCL
# ------------------------------------------------------------

st.subheader("Teorema Central do Limite — TCL")

tamanho_amostra_tcl = st.slider(
    "Tamanho de cada amostra:",
    min_value=2,
    max_value=200,
    value=30,
    step=1,
    key="tcl_amostra"
)

numero_simulacoes = st.slider(
    "Número de amostras:",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100,
    key="tcl_simulacoes"
)

rng_tcl = np.random.default_rng(42)

medias_amostrais = np.mean(
    rng_tcl.choice(
        dados_simulacao,
        size=(
            numero_simulacoes,
            tamanho_amostra_tcl
        ),
        replace=True
    ),
    axis=1
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    medias_amostrais,
    bins=30,
    density=True,
    edgecolor="black"
)

# Curva normal aproximada
media_tcl = calcular_media(dados_simulacao)

desvio_tcl = calcular_desvio_padrao(
    dados_simulacao,
    ddof=1
)

erro_padrao_tcl = (
    desvio_tcl
    / np.sqrt(tamanho_amostra_tcl)
)

x_normal = np.linspace(
    medias_amostrais.min(),
    medias_amostrais.max(),
    300
)

curva_normal = stats.norm.pdf(
    x_normal,
    media_tcl,
    erro_padrao_tcl
)

ax.plot(
    x_normal,
    curva_normal,
    linewidth=2,
    label="Distribuição normal aproximada"
)

ax.set_title(
    "Distribuição das médias amostrais"
)

ax.set_xlabel("Média amostral")
ax.set_ylabel("Densidade")

ax.legend()

st.pyplot(fig)

st.info(
    "O histograma foi construído com density=True para representar "
    "uma densidade de probabilidade. Com amostras suficientemente "
    "grandes, a distribuição das médias tende a apresentar "
    "comportamento aproximadamente normal."
)


# ============================================================
# CRITÉRIO 4
# CORRELAÇÃO E REGRESSÃO
# ============================================================

st.header("📈 4. Correlação e regressão linear")

st.write(
    "Nesta seção são calculadas a correlação de Pearson e a "
    "regressão linear simples pelo Método dos Mínimos Quadrados (MMQ)."
)

variaveis_regressao = [
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

variavel_x = st.selectbox(
    "Variável independente (X):",
    variaveis_regressao,
    key="regressao_x"
)

variavel_y = "cnt"

x = df[variavel_x].values
y = df[variavel_y].values

dados_reg = pd.DataFrame({
    "x": x,
    "y": y
}).dropna()

x = dados_reg["x"].values
y = dados_reg["y"].values

correlacao = calcular_correlacao(x, y)

b0, b1, y_estimado, residuos, r2 = regressao_mmq(
    x,
    y
)

st.subheader("Resultados")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Correlação de Pearson",
        f"{correlacao:.4f}"
    )

with col2:
    st.metric(
        "R²",
        f"{r2:.4f}"
    )

with col3:
    st.metric(
        "Inclinação (b1)",
        f"{b1:.4f}"
    )


st.write(
    f"**Equação da regressão:** "
    f"ŷ = {b0:.4f} + ({b1:.4f}) × {variavel_x}"
)


# Interpretação automática da correlação

valor_abs = abs(correlacao)

if valor_abs < 0.20:
    intensidade = "muito fraca"
elif valor_abs < 0.40:
    intensidade = "fraca"
elif valor_abs < 0.60:
    intensidade = "moderada"
elif valor_abs < 0.80:
    intensidade = "forte"
else:
    intensidade = "muito forte"

if correlacao > 0:
    direcao = "positiva"
elif correlacao < 0:
    direcao = "negativa"
else:
    direcao = "praticamente inexistente"

st.info(
    f"Interpretação: a relação linear entre **{variavel_x}** "
    f"e **{variavel_y}** é {intensidade} e {direcao} "
    f"com base no coeficiente de Pearson."
)


# ------------------------------------------------------------
# GRÁFICO DA REGRESSÃO
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ordem = np.argsort(x)

ax.scatter(
    x,
    y,
    alpha=0.25,
    s=10,
    label="Observações"
)

ax.plot(
    x[ordem],
    y_estimado[ordem],
    linewidth=2,
    label="Regressão MMQ"
)

ax.set_title(
    f"Regressão linear: {variavel_x} × {variavel_y}"
)

ax.set_xlabel(variavel_x)
ax.set_ylabel(variavel_y)

ax.legend()

st.pyplot(fig)


# ------------------------------------------------------------
# PREDIÇÃO COM LIMITES
# ------------------------------------------------------------

st.subheader("🔮 Predição com intervalo")

x_min = float(np.min(x))
x_max = float(np.max(x))

x_novo = st.number_input(
    f"Digite um valor de {variavel_x} para realizar uma previsão:",
    min_value=x_min,
    max_value=x_max,
    value=float((x_min + x_max) / 2)
)

confianca = st.selectbox(
    "Nível de confiança:",
    [0.90, 0.95, 0.99],
    index=1,
    format_func=lambda valor: f"{valor * 100:.0f}%",
    key="confianca_predicao"
)

(
    previsao,
    limite_inferior_pred,
    limite_superior_pred,
    r2_pred
) = intervalo_predicao(
    x,
    y,
    x_novo,
    confianca
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Previsão",
        f"{previsao:.2f}"
    )

with col2:
    st.metric(
        "Limite inferior",
        f"{limite_inferior_pred:.2f}"
    )

with col3:
    st.metric(
        "Limite superior",
        f"{limite_superior_pred:.2f}"
    )

st.info(
    f"Para {variavel_x} = {x_novo:.4f}, a previsão estimada "
    f"para a quantidade de aluguéis é **{previsao:.2f}**, "
    f"com intervalo de predição de {confianca * 100:.0f}% "
    f"entre **{limite_inferior_pred:.2f}** e "
    f"**{limite_superior_pred:.2f}**."
)


# ============================================================
# MATRIZ DE CORRELAÇÃO
# ============================================================

st.subheader("🔗 Matriz de correlação")

variaveis_numericas = df.select_dtypes(
    include=np.number
)

correlacao_matriz = variaveis_numericas.corr()

st.dataframe(
    correlacao_matriz.round(3),
    width="stretch"
)


# ============================================================
# MAPA DE CALOR
# ============================================================

st.subheader("🔥 Mapa de calor da correlação")

fig, ax = plt.subplots(figsize=(12, 8))

imagem = ax.imshow(
    correlacao_matriz,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(
    range(len(correlacao_matriz.columns))
)

ax.set_yticks(
    range(len(correlacao_matriz.columns))
)

ax.set_xticklabels(
    correlacao_matriz.columns,
    rotation=90
)

ax.set_yticklabels(
    correlacao_matriz.columns
)

fig.colorbar(imagem)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# EXPLORAÇÃO INTERATIVA
# ============================================================

st.header("🔎 Exploração interativa")

variavel_exploracao = st.selectbox(
    "Escolha uma variável para comparar com os aluguéis:",
    [
        "temp",
        "atemp",
        "hum",
        "windspeed"
    ],
    key="exploracao_variavel"
)

st.subheader(
    f"Relação entre {variavel_exploracao} "
    "e quantidade de aluguéis"
)

dados_exploracao = df[
    [variavel_exploracao, "cnt"]
].dropna()

st.scatter_chart(
    dados_exploracao,
    x=variavel_exploracao,
    y="cnt",
    width="stretch"
)


# ============================================================
# ANÁLISES POR CATEGORIAS
# ============================================================

st.header("📅 Análises por categorias")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Aluguéis por hora")

    media_hora = (
        df.groupby("hr")["cnt"]
        .mean()
        .reset_index()
    )

    st.line_chart(
        media_hora,
        x="hr",
        y="cnt",
        width="stretch"
    )


with col2:

    st.subheader("Aluguéis por mês")

    media_mes = (
        df.groupby("mnth")["cnt"]
        .mean()
        .reset_index()
    )

    st.bar_chart(
        media_mes,
        x="mnth",
        y="cnt",
        width="stretch"
    )


col1, col2 = st.columns(2)

with col1:

    st.subheader("Aluguéis por estação")

    media_estacao = (
        df.groupby("season")["cnt"]
        .mean()
        .reset_index()
    )

    st.bar_chart(
        media_estacao,
        x="season",
        y="cnt",
        width="stretch"
    )


with col2:

    st.subheader("Aluguéis por condição climática")

    media_clima = (
        df.groupby("weathersit")["cnt"]
        .mean()
        .reset_index()
    )

    st.bar_chart(
        media_clima,
        x="weathersit",
        y="cnt",
        width="stretch"
    )


# ============================================================
# RESUMO AUTOMÁTICO
# ============================================================

st.header("📝 Resumo das análises")

st.write(
    f"""
O conjunto de dados analisado possui **{len(df):,} registros**
e **{len(df.columns)} variáveis**.

A variável de interesse principal é a quantidade de bicicletas
alugadas (`cnt`), que pode ser analisada em função de horário,
mês, estação, condições climáticas, temperatura, umidade e
velocidade do vento.

As estatísticas descritivas apresentam medidas de tendência
central e dispersão, incluindo média, mediana, variância,
desvio-padrão e intervalo interquartil.

A regra de Sturges é utilizada para definir automaticamente
o número de classes do histograma.

O intervalo interquartil permite identificar observações que
podem ser consideradas potenciais outliers pelo critério de
1,5 × IQR.

As simulações da Lei dos Grandes Números mostram a convergência
da média amostral, enquanto o Teorema Central do Limite permite
observar o comportamento da distribuição das médias amostrais.

A análise de correlação mede a associação linear entre as
variáveis. A regressão linear é calculada pelo Método dos
Mínimos Quadrados e apresenta o coeficiente de determinação R²
e uma previsão com intervalo de predição.
"""
)


# ============================================================
# CONCLUSÃO
# ============================================================

st.success(
    "✅ Laboratório Estatístico concluído e pronto para a etapa de documentação e entrega."
)