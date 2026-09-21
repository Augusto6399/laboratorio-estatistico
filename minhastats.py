"""Núcleo estatístico próprio da Sistematização.
Nenhuma medida estatística exibida pela aplicação é calculada com funções prontas.
"""


def _validar_dados(dados):
    if len(dados) == 0:
        raise ValueError("sequência vazia")


def media(dados):
    _validar_dados(dados)
    return sum(dados) / len(dados)


def variancia(dados, amostral=True):
    n = len(dados)
    if n == 0 or (amostral and n < 2):
        raise ValueError("variância amostral exige n >= 2")
    m = media(dados)
    soma_quad = sum((x - m) ** 2 for x in dados)
    return soma_quad / (n - 1 if amostral else n)


def desvio_padrao(dados, amostral=True):
    return variancia(dados, amostral) ** 0.5


def mediana(dados):
    _validar_dados(dados)
    x = sorted(dados)
    n = len(x)
    meio = n // 2
    if n % 2:
        return x[meio]
    return (x[meio - 1] + x[meio]) / 2


def moda(dados):
    _validar_dados(dados)
    contagens = {}
    for valor in dados:
        contagens[valor] = contagens.get(valor, 0) + 1
    maior = max(contagens.values())
    return sorted([v for v, c in contagens.items() if c == maior], key=lambda z: str(z))


def amplitude(dados):
    _validar_dados(dados)
    return max(dados) - min(dados)


def percentil(dados, p):
    _validar_dados(dados)
    if not 0 <= p <= 100:
        raise ValueError("p deve estar entre 0 e 100")
    x = sorted(dados)
    pos = (p / 100) * (len(x) - 1)
    i = int(pos)
    f = pos - i
    if i == len(x) - 1:
        return x[i]
    return x[i] + f * (x[i + 1] - x[i])


def quartis(dados):
    return percentil(dados, 25), percentil(dados, 50), percentil(dados, 75)


def coef_variacao(dados):
    m = media(dados)
    if abs(m) < 1e-12:
        raise ValueError("coeficiente de variação indefinido para média zero")
    return desvio_padrao(dados, True) / m * 100


def covariancia(x, y):
    if len(x) != len(y):
        raise ValueError("vetores devem ter o mesmo tamanho")
    if len(x) < 2:
        raise ValueError("covariância exige pelo menos 2 observações")
    mx, my = media(x), media(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - 1)


def correlacao(x, y):
    if len(x) != len(y):
        raise ValueError("vetores devem ter o mesmo tamanho")
    sx, sy = desvio_padrao(x), desvio_padrao(y)
    if sx == 0 or sy == 0:
        raise ValueError("correlação indefinida para variável constante")
    return covariancia(x, y) / (sx * sy)


def regressao_linear(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x e y devem ter o mesmo tamanho e n >= 2")
    mx, my = media(x), media(y)
    denominador = sum((xi - mx) ** 2 for xi in x)
    if denominador == 0:
        raise ValueError("não é possível regressão com x constante")
    b1 = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / denominador
    b0 = my - b1 * mx
    yhat = [b0 + b1 * xi for xi in x]
    sq_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    sq_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - sq_res / sq_tot if sq_tot else 0.0
    return b0, b1, r2


def sturges(n):
    if n < 1:
        raise ValueError("n deve ser positivo")
    # log10 implementado sem estatística pronta; math.log10 é transformação matemática.
    import math
    return max(1, int(round(1 + 3.322 * math.log10(n))))


def limites_iqr(dados):
    q1, _, q3 = quartis(dados)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr
