# Lab 2 — T8 Customer Personality Analysis

**Problema.** Un lanț de magazine pregătește o nouă campanie de oferte. Marketingul vrea să știe:
- **ce clienți vor accepta oferta** → clasificare, ținta `Response` (0/1);
- **cât cheltuie un client în doi ani** → regresie, ținta = suma coloanelor `Mnt*`.

**Set de date:** [Kaggle — imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), fișierul `marketing_campaign.csv`, 2240 clienți × 29 coloane, separator TAB.

## Rulare

În terminal, din folderul `ML`:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
python lab2\todo1.py     # un singur TODO
python lab2\lab2.py      # toate TODO-urile, în ordine
```

## Structură

```
lab2/
├── lab2.md      # acest fișier: ce s-a făcut și ce s-a găsit la fiecare TODO
├── lab2.py      # rulează toate todoN.py în ordine numerică
├── todo1.py     # TODO 1: importarea datelor            -> load_data()
├── todo2.py     # TODO 2: analiza structurii setului    -> split_columns()
└── data/
    └── raw.csv  # copia locală a setului (separată prin virgulă)
```

Fiecare TODO are fișierul lui și se poate rula separat. TODO-urile următoare refolosesc codul anterior, de ex. `from todo1 import load_data` și `from todo2 import split_columns`.

---

## TODO 1 — Importarea setului de date → `todo1.py`

### Ce am făcut
- `load_data()` citește `lab2/data/raw.csv` dacă există. Altfel, la prima rulare, descarcă setul cu `kagglehub`, citește `marketing_campaign.csv` cu `sep="\t"` și salvează copia locală `raw.csv`, separată prin virgulă. De aceea, la citirea din `raw.csv` nu mai trebuie `sep="\t"`;
- căile sunt calculate față de locația scriptului, deci scriptul merge din orice folder;
- un `assert` verifică forma `(2240, 29)`, apoi se afișează `df.head()`.

### Ce am găsit
- setul are **2240 rânduri × 29 coloane**, exact ca în fișă;
- prima rulare afișează „Downloaded and saved to disk”, iar următoarele „Loaded from disk”.

---

## TODO 2 — Analiza structurii setului de date → `todo2.py`

### Ce am făcut
1. **Separarea coloanelor:** `split_columns(df)` împarte coloanele pe roluri (ID, țintă, dată) și pe tipuri (categoriale, numerice → binare / constante / restul). Un `assert` verifică că grupurile acoperă exact cele 29 de coloane, fiecare o singură dată.
2. **Cardinalitate:** un tabel pe coloană cu dtype, nr. valori distincte, % unice, valori lipsă, valoarea cea mai frecventă și ponderea ei; plus distribuția valorilor categoriale.
3. **Duplicate:** rânduri complet identice, ID-uri repetate, rânduri identice ignorând `ID`.
4. **Coloane speciale:** identificatori, constante, indicatori binari, date stocate ca text, valori lipsă, categorii rare, dezechilibrul țintei.
5. **Observații:** răspunsurile la întrebările laboratorului (mai jos), calculate direct din date.

### Ce am găsit

**Separarea coloanelor**

| Grup | Nr. | Coloane |
|---|---|---|
| ID | 1 | `ID` |
| Țintă clasificare | 1 | `Response` |
| Dată (stocată ca text) | 1 | `Dt_Customer` |
| Categoriale | 2 | `Education` (5 valori), `Marital_Status` (8 valori) |
| Numerice | 24 | 16 cu mai mult de 2 valori, 6 binare (`AcceptedCmp1..5`, `Complain`), 2 constante (`Z_CostContact`, `Z_Revenue`) |
| ↳ părți ale țintei de regresie | 6 | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |

Tipuri: 25 × `int64`, 3 × `str`, 1 × `float64`.

**Cardinalitate**
- 1 valoare: `Z_CostContact`, `Z_Revenue` (constante);
- 2 valori: indicatorii 0/1 și `Response`;
- 3 valori: `Kidhome`, `Teenhome` (0, 1, 2 copii; sunt numere, nu variabile continue);
- 14–16 valori: numărul de cumpărări pe canale și vizitele web pe lună;
- ridicată: `Income` (1974 valori), `MntWines` (776), `Dt_Customer` (663);
- `ID`: 2240 de valori, una pe rând.

**Duplicate**
- 0 rânduri complet identice și 0 ID-uri repetate;
- **182 de rânduri sunt copii ale altora dacă ignorăm `ID`** (358 de rânduri implicate). Sunt probabil același client înregistrat de două ori, cu ID-uri diferite (exemplu: ID 2114 și 4261).

**Altele**
- valori lipsă doar în `Income`: 24 (≈1%);
- ținta `Response` e dezechilibrată: **85,1% 0 și 14,9% 1**. La model se folosesc split stratificat și metrici ca F1 sau ROC-AUC, nu doar acuratețea.

### Observații (răspunsuri la întrebările din laborator)

**a) Ce coloane ar trebui eliminate? Există? De ce?** Da, 3 coloane:

| Coloană | De ce se elimină |
|---|---|
| `ID` | identificator unic pe rând (2240 valori distincte la 2240 de rânduri); nu descrie clientul. Un model l-ar „memora” fără să învețe ceva util |
| `Z_CostContact` | constantă: valoarea 3 la toți clienții, varianță 0, deci nu poate diferenția clienții |
| `Z_Revenue` | constantă: valoarea 11 la toți clienții, același motiv |

Alte coloane nu se șterg din set, dar **nu pot fi feature-uri**: `Mnt*` la regresie (suma lor *este* ținta, deci ar fi scurgere de informație), iar `Response` nu se folosește la regresie.

**b) Ce coloane au tipul greșit?**

| Coloană | Tip actual | Tip corect | Motiv |
|---|---|---|---|
| `Dt_Customer` | `str` | `datetime64` | dată scrisă ca text `zi-lună-an` (2012-07-30 … 2014-06-29) |
| `Education` | `str` | `category` ordinală | are o ordine naturală: Basic < Graduation < 2n Cycle ≈ Master < PhD |
| `Marital_Status` | `str` | `category` | un set fix de 8 etichete, nu text liber |
| `AcceptedCmp1..5`, `Complain`, `Response` | `int64` | `bool` / `category` | indicatori da/nu, nu cantități |
| `Income` | `float64` | ok | e float doar din cauza celor 24 de valori lipsă |
| `Year_Birth` | `int64` | ok | tipul e corect, dar e mai util transformat în vârstă |

**c) Ce coduri ascunse am găsit?**

| Cod | Rânduri | Interpretare |
|---|---|---|
| `Marital_Status` = `Absurd`, `YOLO` | 4 | nu sunt stări civile, ci valori de umplutură sau glume la completare. Cele 2 `YOLO` (ID 492, 11133) sunt și duplicate între ele |
| `Marital_Status` = `Alone` | 3 | sinonim pentru `Single` → de unificat |
| `Income` = 666666 | 1 (ID 9432) | cod din cifre repetate, de tip „nu știu”. Următorul venit ca mărime e 162397, deci nu e un venit real |
| `Year_Birth` = 1893, 1899, 1900 | 3 (ID 11004, 1150, 7829) | vârste de 114–121 de ani în 2014; anul a fost completat greșit sau pus ca valoare implicită |
| `Income` = NaN | 24 | valori lipsă explicite |
| `Education` = `2n Cycle` | 203 | etichetă din sistemul Bologna, echivalentă cu `Master` → de decis dacă se unifică |

Inconsistențe, posibile erori de înregistrare:
- 6 clienți au 0 cumpărări (web + catalog + magazin), dar au cheltuit bani (ID 5555, 3955, 6862, 11110, 9931, 11181);
- 3 clienți au mai multe cumpărări cu reducere decât cumpărări în total (ID 10749, 6862, 9931).

### De făcut la curățare (TODO-uri viitoare)
- eliminarea coloanelor `ID`, `Z_CostContact`, `Z_Revenue`;
- conversia `Dt_Customer` la dată (eventual vechimea clientului) și a `Year_Birth` la vârstă;
- tratarea celor 24 de valori lipsă din `Income` și a valorii 666666;
- tratarea anilor de naștere imposibili (1893, 1899, 1900);
- gruparea `Alone` / `Absurd` / `YOLO` în `Marital_Status` și decizia pentru `2n Cycle`;
- decizia pentru cele 182 de duplicate ignorând `ID`;
- coloanele `Mnt*` nu intră ca feature-uri la regresie.
