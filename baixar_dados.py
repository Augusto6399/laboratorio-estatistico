"""Baixa o Bike Sharing diretamente do UCI e grava dados/dataset.csv."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dados" / "dataset.csv"
OUT.parent.mkdir(exist_ok=True)

bike = fetch_ucirepo(id=275)
df = pd.concat([bike.data.features, bike.data.targets], axis=1)
# A variável hr existe no conjunto horário e é tratada como categórica na análise.
df.to_csv(OUT, index=False)
print(f"Dataset salvo em: {OUT}")
print(f"Dimensões: {df.shape}")
print(df.dtypes)
