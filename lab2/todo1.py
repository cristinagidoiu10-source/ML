"""TODO 1 — Importarea setului de date (T8 Customer Personality Analysis).

Citeste setul din lab2/data/raw.csv; la prima rulare il descarca de pe Kaggle
si salveaza copia locala. Dimensiune asteptata: 2240 clienti x 29 coloane.
Sursa: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Rulare (cu .venv activat, din folderul ML):
    python lab2\\todo1.py

Celelalte TODO-uri refolosesc datele cu:
    from todo1 import load_data
"""

import glob
import os

import pandas as pd

# Toate fisierele lab-ului stau langa acest script, in lab2/data
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LAB_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load_data():
    """Returneaza setul ca DataFrame (de pe disc sau, prima data, de pe Kaggle)."""
    raw_csv = os.path.join(DATA_DIR, "raw.csv")      # copia locala a setului

    if os.path.exists(raw_csv):                      # exista deja pe disc? -> il citim de acolo
        df = pd.read_csv(raw_csv)
        print("Loaded from disk")
    else:                                            # prima rulare -> il descarcam o singura data si il salvam
        import kagglehub
        p = kagglehub.dataset_download("imakash3011/customer-personality-analysis")  # p = folderul descarcat
        f = glob.glob(p + "/**/marketing_campaign.csv", recursive=True)[0]         # cautam fisierul si in subfoldere
        df = pd.read_csv(f, sep="\t")                # fisierul original foloseste TAB ca separator
        df.to_csv(raw_csv, index=False)              # copia locala e un CSV simplu, separat prin virgula
        print("Downloaded and saved to disk")

    return df


if __name__ == "__main__":
    df = load_data()

    print(df.shape)                                  # (randuri, coloane): comparati cu fisa setului
    assert df.shape == (2240, 29), f"Forma neasteptata: {df.shape} (fisa: 2240 x 29)"
    print(df.head())
