# Laboratório Estatístico Interativo

Projeto da Sistematização de Matemática e Estatística para Computação — 2026.

## Dataset
**Bike Sharing**, UCI Machine Learning Repository (ID 275), com dados horários e diários de aluguel de bicicletas no sistema Capital Bikeshare entre 2011 e 2012.

Fonte original: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
DOI: https://doi.org/10.24432/C5W894

## Estrutura
- `app.py` — interface Streamlit.
- `minhastats.py` — núcleo estatístico próprio.
- `test_minhastats.py` — testes de validação contra NumPy.
- `baixar_dados.py` — baixa o dataset original e cria `dados/dataset.csv`.
- `gerar_relatorio.py` — gera o relatório final depois que os dados forem baixados.
- `dados/` — dataset local.

## Como executar no VS Code / terminal
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python baixar_dados.py
pytest -v
streamlit run app.py
```

Depois, em outro terminal, o relatório pode ser gerado com:
```bash
python gerar_relatorio.py
```

## Regra de ouro
A aplicação não usa `np.mean`, `np.var`, `np.median` etc. para produzir as medidas mostradas ao usuário. Essas funções ficam restritas aos testes de validação.

## Integrantes
Preencher no relatório final: 
NOME: AUGUSTO DA SILVA CARNEIRO DE ARRUDA 
Matrícula: 72650316 
NOME: Bruno Seabre Lima de Moura
Matrícula: 72650566
