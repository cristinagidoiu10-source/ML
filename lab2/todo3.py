"""TODO 3 — Statistici descriptive, corelatii si verificarea leakage-ului (E - Explore).

Acelasi TODO ca in repo-ul de referinta (Telco), facut pe setul nostru:
T8 Customer Personality Analysis.
  - clasificare: Response (a acceptat oferta? 0/1)
  - regresie:    MntTotal  (suma celor 6 coloane Mnt* = cat a cheltuit clientul in 2 ani)

Rulare (cu .venv activat, din folderul ML):
    python lab2\\todo3.py
"""

import numpy as np
import pandas as pd

from todo1 import load_data
from todo2 import ID_COL, TARGET_CLS

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = load_data()

# --- pregatire minima (NU e curatarea finala, doar ce ne trebuie ca sa calculam statisticile) ---
spend_cols = [c for c in df.columns if c.startswith("Mnt")]
df["MntTotal"] = df[spend_cols].sum(axis=1)                      # tinta de regresie
df["Age"] = 2014 - df["Year_Birth"]                              # varsta in 2014 (ultimul an din date)
dt = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")        # text "zi-luna-an" -> data
df["Customer_Days"] = (dt.max() - dt).dt.days                    # de cate zile e client (fata de ultima inscriere)

TARGET = TARGET_CLS                                              # "Response"
TARGET_REG = "MntTotal"
num_cols = ["Age", "Income", "Kidhome", "Teenhome", "Recency", "Customer_Days", *spend_cols, TARGET_REG,
            "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
            "NumWebVisitsMonth"]
cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain"]
cat_cols = ["Education", "Marital_Status", "Kidhome", "Teenhome", *cmp_cols]   # pentru ratele pe categorii
y = df[TARGET]                                                   # deja 0/1

# 1. describe(): coloane numerice si categoriale
print("\n=== describe() numerice ===")
print(df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1).T)
print("\n=== describe() categoriale ===")
print(df[["Education", "Marital_Status"]].describe())          # count, unique, top (= modul), freq


# 2. Tabel cu metrici per coloana numerica
def numeric_summary(data):
    rows = {}
    for c in data.select_dtypes(include="number").columns:
        s = data[c]
        q1, q3 = s.quantile([0.25, 0.75])                                 # cuartilele 1 si 3
        iqr = q3 - q1                                                     # latimea jumatatii din mijloc
        outside = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()     # candidati la outlier (IQR)
        rows[c] = {"missing %": 100 * s.isna().mean(), "mean": s.mean(), "median": s.median(),
                   "std": s.std(), "CV": s.std() / s.mean() if s.mean() != 0 else np.nan,
                   "min": s.min(), "max": s.max(), "IQR": iqr, "IQR outliers": outside,
                   "skew": s.skew(), "kurtosis": s.kurt()}
    return pd.DataFrame(rows).T.round(2)                                  # un rand pentru fiecare coloana


print("\n=== Metrici per coloana numerica ===")
print(numeric_summary(df[num_cols]))

# 3. Valori lipsa, distributia tintei, statistici per clasa
print("\n=== Valori lipsa (%) ===")
missing = (100 * df.isna().mean()).round(2).sort_values(ascending=False)
print(missing[missing > 0])
print(f"({df.isna().sum().sum()} valori lipsa din {df.size} celule)")

counts = y.value_counts()
print("\n=== Distributia tintei ===")
print(y.value_counts(normalize=True).round(3))
# raportul de dezechilibru: de cate ori este mai mare clasa cea mai frecventa decat cea mai rara
print("imbalance ratio (largest / smallest class):", round(counts.max() / counts.min(), 2))

print("\n=== Media si mediana coloanelor numerice, per clasa ===")
print(df.groupby(TARGET)[["Income", "Recency", "Customer_Days", "MntWines", "MntTotal",
                          "NumCatalogPurchases", "Age"]].agg(["mean", "median"]).round(1).T)

# pentru fiecare coloana categoriala: rata de raspuns (% Response = 1) in fiecare categorie
# diferenta mare intre categorii (spread) = atribut legat de tinta
print("\n=== Rata de raspuns (%) pe categorii ===")
spread = {}
for c in cat_cols:
    rate = (100 * y.groupby(df[c]).mean()).round(1)
    spread[c] = rate.max() - rate.min()
    print(f"{c:15s} {rate.to_dict()}")
print("\nCele mai legate de tinta (spread = max - min rata de raspuns, in puncte procentuale):")
print(pd.Series(spread).sort_values(ascending=False).round(1))

# tabel de contingenta pentru cel mai puternic atribut categorial
print("\n=== AcceptedCmp5 x Response (numar de clienti) ===")
print(pd.crosstab(df["AcceptedCmp5"], df[TARGET], margins=True))

# 4. Corelatii Pearson si Spearman (numerice + tinta 0/1)
corr_df = df[num_cols + cmp_cols].assign(**{TARGET: y})
pearson = corr_df.corr(method="pearson")      # legatura liniara (pe o dreapta)
spearman = corr_df.corr(method="spearman")    # legatura monotona (ordinea valorilor)

# cele mai puternic corelate perechi de atribute (fara tinte), fiecare pereche o singura data
attrs = [c for c in num_cols if c != TARGET_REG]
p_attr = pearson.loc[attrs, attrs]
pairs = p_attr.where(np.triu(np.ones(p_attr.shape, dtype=bool), k=1)).stack()
print("\n=== Cele mai corelate perechi de atribute (Pearson) ===")
print(pairs.abs().sort_values(ascending=False).head(10).round(2))

print(f"\n=== Corelatia cu {TARGET} (Spearman) ===")
print(spearman[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(2))

print(f"\n=== Corelatia cu {TARGET_REG} (Spearman), fara coloanele Mnt* ===")
print(spearman[TARGET_REG].drop([TARGET_REG, TARGET, *spend_cols]).sort_values(key=abs, ascending=False).round(2))

# 5. Verificare leakage
print("\n=== Leakage ===")
# clasificare: o corelatie atribut-tinta ~ +-1 sau o categorie cu raspuns 0% / 100% ar fi suspecta
print("max |corelatie| atribut-Response:", spearman[TARGET].drop(TARGET).abs().max().round(2))
pure = [(c, v, int((df[c] == v).sum())) for c in cat_cols
        for v, r in y.groupby(df[c]).mean().items() if r in (0, 1)]
print("categorii cu raspuns 0% sau 100% (coloana, valoare, nr. clienti):", pure)
# regresie pe MntTotal: suma coloanelor Mnt* ESTE tinta -> corelatie 1
print("corr(MntTotal, suma Mnt*) =", round(df[TARGET_REG].corr(df[spend_cols].sum(axis=1)), 4))
purch = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
print("corr(MntTotal, nr. total de cumparari) =", round(df[TARGET_REG].corr(purch, method="spearman"), 2),
      "-> numarate in aceeasi perioada de 2 ani ca tinta")
