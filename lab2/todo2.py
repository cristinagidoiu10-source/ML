"""TODO 2 — Analiza structurii setului de date.

  1. Separarea coloanelor: pe roluri (ID, tinte, data) si pe tipuri (numerice / categoriale).
  2. Cardinalitate, duplicate si coloane speciale.

Rulare (cu .venv activat, din folderul ML):
    python lab2\\todo2.py

Celelalte TODO-uri pot refolosi separarea cu:
    from todo2 import split_columns
"""

import pandas as pd

from todo1 import load_data

ID_COL = "ID"                       # identificator, nu e feature
TARGET_CLS = "Response"             # tinta de clasificare: a acceptat oferta (0/1)
DATE_COLS = ["Dt_Customer"]         # data inscrierii, stocata ca text "zi-luna-an"


def split_columns(df):
    """Imparte coloanele pe roluri si tipuri; intoarce un dict {grup: [coloane]}."""
    spend = [c for c in df.columns if c.startswith("Mnt")]   # suma lor = tinta de regresie
    special = [ID_COL, TARGET_CLS] + DATE_COLS
    rest = [c for c in df.columns if c not in special]

    numeric = df[rest].select_dtypes(include="number").columns.tolist()
    categorical = df[rest].select_dtypes(exclude="number").columns.tolist()

    nunique = df[numeric].nunique()
    constant = nunique[nunique == 1].index.tolist()             # aceeasi valoare peste tot
    binary = nunique[nunique == 2].index.tolist()               # indicatori 0/1
    nonbinary = [c for c in numeric if c not in constant + binary]   # numerice cu > 2 valori

    return {
        "id": [ID_COL],
        "target_cls": [TARGET_CLS],
        "target_reg_parts": spend,
        "date": DATE_COLS,
        "categorical": categorical,
        "numeric": numeric,                 # toate numericele (fara ID / tinta)
        "numeric_nonbinary": nonbinary,     # numerice cu > 2 valori
        "binary": binary,
        "constant": constant,
    }


def section(title):
    print(f"\n{'-' * 70}\n{title}\n{'-' * 70}")


if __name__ == "__main__":
    df = load_data()
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Separarea coloanelor
    # ------------------------------------------------------------------
    section("1. Tipuri de date (dtype)")
    print(df.dtypes.astype(str).value_counts().to_string())

    section("1. Separarea coloanelor pe grupuri")
    groups = split_columns(df)
    for name, cols in groups.items():
        print(f"{name:20s} ({len(cols):2d}): {cols}")

    # verificare: grupurile de baza acopera toate cele 29 de coloane, fara suprapuneri
    base = groups["id"] + groups["target_cls"] + groups["date"] + groups["categorical"] + groups["numeric"]
    assert sorted(base) == sorted(df.columns), "Separarea nu acopera exact toate coloanele"
    print(f"\nOK: id + tinta + data + categoriale + numerice = {len(base)} coloane (toate, o singura data)")

    # ------------------------------------------------------------------
    # 2. Cardinalitate
    # ------------------------------------------------------------------
    section("2. Cardinalitate (nr. valori distincte pe coloana)")
    card = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "unique_%": (df.nunique() / n * 100).round(2),
        "missing": df.isna().sum(),
        "top_value": df.mode().iloc[0],
        "top_%": (df.apply(lambda s: s.value_counts(normalize=True).iloc[0]) * 100).round(1),
    }).sort_values("n_unique")
    print(card.to_string())

    section("2. Categoriale: distributia valorilor")
    for c in groups["categorical"]:
        print(f"\n{c}:")
        print(df[c].value_counts().to_string())

    # ------------------------------------------------------------------
    # 3. Duplicate
    # ------------------------------------------------------------------
    section("3. Duplicate")
    dup_rows = df.duplicated().sum()
    dup_ids = df[ID_COL].duplicated().sum()
    dup_no_id_mask = df.drop(columns=ID_COL).duplicated(keep=False)
    print(f"Randuri identice complet:            {dup_rows}")
    print(f"ID-uri repetate:                     {dup_ids}")
    print(f"Randuri identice ignorand ID-ul:     {df.drop(columns=ID_COL).duplicated().sum()} "
          f"(copii in plus; {dup_no_id_mask.sum()} randuri implicate)")
    print("\nExemplu: un grup de randuri care difera doar prin ID")
    first = df[dup_no_id_mask].drop(columns=ID_COL).iloc[0]
    print(df[(df.drop(columns=ID_COL) == first).all(axis=1)].to_string())

    # ------------------------------------------------------------------
    # 4. Coloane speciale
    # ------------------------------------------------------------------
    section("4. Coloane speciale")
    id_like = card.index[card["n_unique"] == n].tolist()
    print(f"Identificatori (o valoare unica pe rand):     {id_like}")
    print(f"Constante (o singura valoare -> de eliminat): "
          f"{ {c: df[c].iloc[0].item() for c in groups['constant']} }")
    print(f"Binare 0/1 (indicatori):                      {groups['binary']}")
    print(f"Data stocata ca text (de convertit):          {DATE_COLS} "
          f"-> ex. {df[DATE_COLS[0]].iloc[0]!r}, dtype {df[DATE_COLS[0]].dtype}")
    print(f"Coloane cu valori lipsa:                      {card.index[card['missing'] > 0].tolist()} "
          f"({card.loc[card['missing'] > 0, 'missing'].to_dict()})")

    print("\nCategorii rare (< 1% din randuri):")
    for c in groups["categorical"]:
        share = df[c].value_counts(normalize=True)
        rare = share[share < 0.01]
        print(f"  {c}: {(rare * n).round().astype(int).to_dict()}")

    print(f"\nRata tintei {TARGET_CLS}: {df[TARGET_CLS].value_counts(normalize=True).round(3).to_dict()} "
          f"(clase dezechilibrate)")

    # ------------------------------------------------------------------
    # 5. Observatii (TODO): raspunsurile la intrebarile din laborator
    # ------------------------------------------------------------------
    section("5a. Ce coloane ar trebui eliminate? (si de ce)")
    drop = {
        ID_COL: "identificator unic pe rand (2240/2240 valori) -> nu spune nimic despre client",
        **{c: f"constanta (= {df[c].iloc[0].item()} pentru toti) -> varianta 0, nu ajuta modelul"
           for c in groups["constant"]},
    }
    for c, why in drop.items():
        print(f"  {c:15s} {why}")
    print("  Atentie (nu se elimina din set, dar NU intra ca feature):")
    print(f"    - {groups['target_reg_parts']} pentru regresie: suma lor E tinta -> scurgere (leakage)")
    print(f"    - {TARGET_CLS} pentru regresie si invers: e tinta celeilalte probleme")

    section("5b. Ce coloane au tipul gresit?")
    dt = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    wrong_types = [
        ("Dt_Customer", str(df["Dt_Customer"].dtype), "datetime64",
         f"data e text 'zi-luna-an' (interval {dt.min():%Y-%m-%d} .. {dt.max():%Y-%m-%d})"),
        ("Education", str(df["Education"].dtype), "category (ordinala)",
         "nivel de studii, are ordine: Basic < Graduation < 2n Cycle ~ Master < PhD"),
        ("Marital_Status", str(df["Marital_Status"].dtype), "category", "8 etichete fixe, nu text liber"),
        ("AcceptedCmp1..5, Complain, Response", "int64", "bool / category",
         "indicatori 0/1, nu cantitati"),
        ("Income", str(df["Income"].dtype), "float64 (ok)",
         "e float doar din cauza celor 24 NaN; tipul e corect"),
        ("Year_Birth", "int64", "int64 (ok) -> Age",
         "tip corect, dar mai util transformat in varsta"),
    ]
    for col, now, should, why in wrong_types:
        print(f"  {col:38s} {now:8s} -> {should:22s} {why}")

    section("5c. Ce coduri ascunse am gasit?")
    hidden = {
        "Marital_Status = Absurd / YOLO": df["Marital_Status"].isin(["Absurd", "YOLO"]),
        "Marital_Status = Alone (sinonim Single)": df["Marital_Status"].eq("Alone"),
        "Income = 666666 (cod 'cifre repetate')": df["Income"].eq(666666),
        "Year_Birth < 1920 (varsta > 94 ani)": df["Year_Birth"].lt(1920),
        "Income = NaN (lipsa)": df["Income"].isna(),
    }
    for label, mask in hidden.items():
        print(f"  {label:42s} {mask.sum():3d} randuri, ID-uri: {df.loc[mask, ID_COL].tolist()[:8]}")
    print(f"  {'Education = 2n Cycle':42s} {df['Education'].eq('2n Cycle').sum():3d} randuri "
          f"(echivalent Master in sistemul Bologna -> de verificat daca se unifica)")

    purchases = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
    spend_total = df[groups["target_reg_parts"]].sum(axis=1)
    incons = {
        "0 cumparari, dar cheltuiala > 0": (purchases == 0) & (spend_total > 0),
        "NumDealsPurchases > total cumparari": df["NumDealsPurchases"] > purchases,
    }
    print("\n  Inconsistente (posibile coduri / erori de inregistrare):")
    for label, mask in incons.items():
        print(f"  {label:42s} {mask.sum():3d} randuri, ID-uri: {df.loc[mask, ID_COL].tolist()}")
