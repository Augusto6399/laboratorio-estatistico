from pathlib import Path
import pandas as pd
import minhastats as ms

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dados' / 'dataset.csv'
OUT = ROOT / 'RELATORIO.md'

df = pd.read_csv(DATA)
num = [c for c in df.select_dtypes(include='number').columns if c != 'instant']

# Estatísticas usadas no relatório
cnt = df['cnt'].dropna().tolist()
temp = df['temp'].dropna().tolist()
r_temp_cnt = ms.correlacao(temp, cnt)
b0,b1,r2 = ms.regressao_linear(temp,cnt)
season = df.groupby('season')['cnt'].median().sort_values(ascending=False)
lo,hi = ms.limites_iqr(cnt)
out = df[(df.cnt < lo) | (df.cnt > hi)]

text = f'''# RELATÓRIO — Laboratório Estatístico Interativo

## 1. Equipe
- Integrantes: **PREENCHER NOMES E MATRÍCULAS**

## 2. Dataset e justificativa
Foi utilizado o dataset **Bike Sharing**, disponibilizado pelo UCI Machine Learning Repository. O conjunto reúne registros de aluguel de bicicletas e informações de horário, clima, estação e calendário. O arquivo utilizado possui **{len(df)} registros** e {len(df.columns)} variáveis.

Fonte original: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset  
DOI: https://doi.org/10.24432/C5W894

A escolha atende aos requisitos do projeto por possuir mais de 1.000 registros, diversas variáveis numéricas e variáveis categóricas. O tratamento não precisou remover linhas por valores ausentes, pois a fonte informa ausência de valores faltantes.

## 3. Tratamento dos dados
Os dados foram carregados com Pandas. A coluna de data foi preservada como informação temporal e as variáveis numéricas foram convertidas para listas antes de serem enviadas ao núcleo `minhastats.py`. Não foram aplicadas imputações.

## 4. Núcleo estatístico próprio
As principais funções implementadas foram:

- Média: $\\bar x = \\frac{{1}}{{n}}\\sum x_i$.
- Variância amostral: $s^2 = \\frac{{\\sum(x_i-\\bar x)^2}}{{n-1}}$.
- Variância populacional: $\\sigma^2 = \\frac{{\\sum(x_i-\\bar x)^2}}{{n}}$.
- Desvio padrão: raiz quadrada da variância.
- Mediana: elemento central após ordenação, ou média dos dois centrais.
- Percentil: posição $p(n-1)/100$ com interpolação linear.
- Coeficiente de variação: $CV = s/\\bar x \\times 100$.
- Covariância e correlação de Pearson.
- Regressão linear por mínimos quadrados e $R^2$.

As funções estatísticas são calculadas pelo próprio código. NumPy aparece somente nos testes de validação, conforme a regra de ouro da atividade.

## 5. Validação
Os testes em `test_minhastats.py` comparam média, variâncias, mediana, percentis, covariância, correlação e regressão com referências NumPy, usando tolerância numérica.

## 6. Descritiva interativa
A aplicação permite selecionar variáveis numéricas e apresenta média, mediana, desvio padrão, quartis, IQR, limites de outliers, histograma e boxplot. Para variáveis categóricas, apresenta tabela/contagem e gráfico de barras.

Para o número de classes do histograma foi usada a regra de Sturges: $k=1+3,322\\log_{{10}}(n)$. Os outliers são identificados pela regra $[Q1-1,5IQR, Q3+1,5IQR]$.

## 7. Simulação e distribuições
A aplicação apresenta a Lei dos Grandes Números por meio de lançamentos simulados de moeda. Também realiza o experimento do Teorema Central do Limite usando amostras da variável escolhida e histogramando suas médias.

O módulo de distribuições permite comparar o histograma dos dados com Normal, Exponencial ou Uniforme, com parâmetros estimados a partir dos próprios dados.

## 8. Correlação e regressão
Para a relação entre temperatura normalizada (`temp`) e total de bicicletas alugadas (`cnt`), os valores calculados foram:

- Correlação de Pearson: **{r_temp_cnt:.4f}**
- Intercepto $b_0$: **{b0:.4f}**
- Inclinação $b_1$: **{b1:.4f}**
- $R^2$: **{r2:.4f}**

A interpretação é associativa: cada unidade adicional de `temp` está associada, em média, a aproximadamente **{b1:.2f}** unidades de `cnt`. Correlação não implica causalidade.

## 9. As três descobertas
### Descoberta 1 — contraste entre estações
As medianas de `cnt` por estação foram:

{season.to_string()}

Esse contraste mostra que a demanda não apresenta a mesma distribuição em todas as estações. A comparação é descritiva e não permite, sozinha, atribuir causalidade.

### Descoberta 2 — temperatura e demanda
A correlação entre `temp` e `cnt` foi **{r_temp_cnt:.4f}**. O resultado indica uma associação linear entre as duas variáveis no conjunto analisado, mas não demonstra que a temperatura seja a causa da variação na demanda.

### Descoberta 3 — pontos fora do IQR
A regra do IQR encontrou **{len(out)} registros** fora dos limites **{lo:.2f}** e **{hi:.2f}** para `cnt`. Esses registros são pontos que merecem investigação e não devem ser automaticamente excluídos, pois podem representar situações reais de alta ou baixa demanda.

## 10. Repositório e vídeo
- Repositório público: **PREENCHER LINK DO GITHUB/GITLAB**
- Vídeo de 3–5 minutos: **PREENCHER LINK**

## 11. Conclusão
O laboratório integra cálculo estatístico implementado manualmente, validação numérica, visualização, simulação, comparação com distribuições teóricas e regressão linear em uma única aplicação interativa. Dessa forma, os conceitos estudados são apresentados como software verificável e reproduzível.
'''
OUT.write_text(text, encoding='utf-8')
print(f'Relatório gerado em {OUT}')
