import numpy as np
import minhastats as ms

DADOS = np.random.default_rng(0).gamma(2, 9, 500).tolist()


def test_media():
    assert np.isclose(ms.media(DADOS), np.mean(DADOS), rtol=1e-9)


def test_variancia_amostral():
    assert np.isclose(ms.variancia(DADOS, True), np.var(DADOS, ddof=1), rtol=1e-9)


def test_variancia_populacional():
    assert np.isclose(ms.variancia(DADOS, False), np.var(DADOS, ddof=0), rtol=1e-9)


def test_mediana():
    assert np.isclose(ms.mediana(DADOS), np.median(DADOS), rtol=1e-9)


def test_percentis():
    for p in [0, 25, 50, 75, 90, 100]:
        assert np.isclose(ms.percentil(DADOS, p), np.percentile(DADOS, p), rtol=1e-6)


def test_covariancia():
    y = [2*x + 3 for x in DADOS]
    assert np.isclose(ms.covariancia(DADOS, y), np.cov(DADOS, y, ddof=1)[0, 1], rtol=1e-9)


def test_correlacao():
    y = [2*x + 3 for x in DADOS]
    assert np.isclose(ms.correlacao(DADOS, y), np.corrcoef(DADOS, y)[0, 1], rtol=1e-9)


def test_regressao():
    x = list(range(1, 11))
    y = [5 + 2*xv for xv in x]
    b0, b1, r2 = ms.regressao_linear(x, y)
    assert np.isclose(b0, 5)
    assert np.isclose(b1, 2)
    assert np.isclose(r2, 1)
