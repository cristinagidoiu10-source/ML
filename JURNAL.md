# Jurnal de lucru — ML

Fiecare sarcină primită e documentată aici: ce s-a cerut, ce s-a făcut, unde se află și ce a rămas de făcut.

---

## T8 — Customer Personality Analysis

**Problema:** un lanț de magazine pregătește o campanie de oferte. Marketingul vrea să știe:
- ce clienți vor accepta oferta → **clasificare** (ținta `Response`);
- cât cheltuie un client în doi ani → **regresie** (ținta = suma coloanelor `Mnt*`).

**Set de date:** [Kaggle — imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), fișierul `marketing_campaign.csv`, 2.240 clienți × 29 coloane, separator TAB.

**Cod:** în `lab2/` (vezi mai jos). Notebook-ul inițial (`T8_Customer_Personality.ipynb`) și folderul `data/` din rădăcină au fost șterse pe 2026-10-12, la cerere. Au fost înlocuite de `lab2/todo1.py` și `lab2/data/`.

### Pas 1 — Importarea setului de date (2026-10-12)

**Cerință:** încărcarea setului în Colab cu copie în Google Drive.

**Ce s-a făcut:**
- creat notebook-ul cu:
  - celula de configurare (montare Drive, `DATA_DIR`, `MY_NAME`, `MY_DIR`);
  - celula de import primită de la laborator: citește `raw.csv` din Drive dacă există, altfel descarcă setul cu `kagglehub`, îl citește cu `sep="\t"` și îl salvează în Drive ca CSV separat prin virgulă;
  - o celulă de verificare: `assert` pe forma (2240, 29), `df.info()`, valori lipsă, ID-uri duplicate;
  - dicționarul celor 29 de coloane, grupate pe categorii.

**De reținut:**
- celula de configurare e o variantă proprie; dacă laboratorul o are pe a lui, se înlocuiește (contează doar ca `MY_DIR` să existe). Trebuie completat `MY_NAME`;
- pe acest calculator nu e instalat Python, așa că notebook-ul nu a fost rulat local; se rulează în Colab (File → Upload notebook);
- prima rulare are nevoie de acces la Kaggle prin `kagglehub` (seturile publice se descarcă de obicei fără login).

### Pas 1b — Mutare de pe Colab pe local (2026-10-12)

**Cerință:** „fără Colab, vreau totul local”.

**Ce s-a făcut:**
- instalat Python 3.12.10 (winget, doar pentru utilizatorul curent);
- creat mediul virtual `.venv` în proiect, cu `pandas`, `kagglehub`, `jupyter` și `ipykernel`;
- descărcat setul cu `kagglehub` în `data/marketing_campaign.csv` (original, separat prin TAB) și convertit în `data/raw.csv` (separat prin virgulă). Ambele fișiere au **2240 rânduri × 29 coloane**, ca în fișă;
- notebook-ul a fost adaptat: fără Drive/Colab, `DATA_DIR = "data"`, citește `data/raw.csv`.

**Blocaj:** Windows **Smart App Control** este activ și blochează bibliotecile compilate fără semnătură (`numpy`, deci și `pandas`, ulterior `scikit-learn`, `matplotlib`): *„An Application Control policy has blocked this file”*. Python și `kagglehub` (cod Python pur) merg, dar notebook-ul nu poate rula până nu se rezolvă blocajul. Există două variante:
1. dezactivarea Smart App Control (Windows Security → App & browser control → Smart App Control → Off);
2. lucrul în WSL (Ubuntu), care nu e afectat de Smart App Control (necesită drepturi de administrator și o repornire).

**Rezolvare:** utilizatorul a oprit Smart App Control; `numpy` 2.5.4 și `pandas` 3.0.6 se importă acum normal.

---

## Lab 2 — folderul `lab2/`

Lucrul continuă ca scripturi Python în `lab2/`, câte un fișier `todoN.py` pentru fiecare TODO. `lab2.py` le rulează pe toate, în ordine.

**Documentația detaliată (ce s-a făcut și ce s-a găsit la fiecare TODO) este în [`lab2/lab2.md`](lab2/lab2.md).**

| TODO | Fișier | Pe scurt | Data |
|---|---|---|---|
| 1 | `todo1.py` | importarea setului (2240 × 29) | 2026-10-12 |
| 2 | `todo2.py` | structura setului: separarea coloanelor, cardinalitate, duplicate, coloane speciale + observații | 2026-10-12 |