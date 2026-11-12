# Raport L02 — Explorarea, curățarea și primele modele (T8 Customer Personality Analysis)

Analiză pe fiecare TODO din `L02_Explorare_Modele (1).ipynb`. Pentru fiecare: **verdict**, **ce e greșit și de ce**, **codul de înlocuit** (din notebook-ul tău) și **codul corect** (copiezi toată celula).

Legendă: ✅ corect · ⚠️ merge, dar are probleme de logică / curățenie · ❌ greșit sau necompletat

> Cifrele din acest raport le-am obținut rulând codul corectat pe `lab2/data/raw.csv` (scikit-learn 1.9). Pe Colab pot ieși mici diferențe la a treia zecimală. Dacă aplici curățarea corectată (TODO 5), **toate scorurile se schimbă puțin**, pentru că setul are acum 2058 de rânduri, nu 2054. Rulează din nou notebook-ul de sus până jos și copiază în observații cifrele care îți ies ție.

## Rezumat

| # | TODO | Verdict | Problema principală |
|---|---|---|---|
| 0 | Setul de date și problema | ❌ | `TARGET_R = ""`, spațiu la finalul `MY_NAME`, lipsește celula de text cu povestea |
| 1 | Încărcarea datelor | ⚠️ | nu salvează `raw.csv` în `MY_DIR`, variabila se numește `customer`, nu `df` |
| 2 | Structura tabelului | ⚠️ | `df.sample` nu se afișează, `if 'df' not in locals()` e fragil |
| 2 | Observații structură | ❌ | `Year_Birth` nu e cod ascuns; ai ratat `666666`, `Absurd`/`YOLO`, duplicatele ascunse de `ID` |
| 2 | Celula AI | ❌ | e copiată de la alt set de date (Obesity, UCI 544) |
| 3 | Întrebarea Penguins | ❌ | necompletată |
| 3 | Statistici descriptive | ⚠️ | **modifică `df` brut** (adaugă `MntTotal`) |
| 3 | Observații statistici | ❌ | răspunsuri incomplete sau greșite (24 de lipsuri, nu una) |
| 4 | Grafice | ✅ / ⚠️ | corecte; doar modifică `df` și au comentarii lăsate de AI |
| 5 | Lista de decizii | ❌ | necompletată |
| 5 | Curățarea | ⚠️ | `666666` ar trebui să devină `NaN`, `2n Cycle` = Master, dată de referință fixă |
| 6 | Întrebări Penguins (arbore, MAE) | ❌ | necompletate |
| 6 | Primele modele | ⚠️ | bloc duplicat, tabelele nu se afișează, nume de variabile luate din exemplu, *leakage* la regresie |
| 6 | Observații modele | ❌ | necompletate |
| 7 | Final | ❌ | lipsește `raw.csv` din `MY_DIR` |

---

## TODO 0 — Setul de date și problema (celula 5) ❌

### Ce e greșit
1. **`TARGET_R = ""`**: ținta de regresie nu e completată, deși o folosești mai jos (`MntTotal`). Mai târziu o redefinești ca `target_r` / `TARGET_R`. Constanta trebuie definită **o singură dată**, sus, și apoi folosită peste tot.
2. **`MY_NAME = "Customer Personality Analysis "`**: are un **spațiu la final**. Numele intră în căi de fișiere: în output se vede `L02_Customer Personality Analysis _first_models.csv`. În L03 ai scris numele fără spațiu, deci cele două laboratoare caută în **foldere diferite**. Laboratorul cere explicit ca `MY_NAME` să fie identic.
3. Lipsește celula de text cerută: detaliile setului, o linie `---` și propoziția *„Prezic X pentru Y, ca să Z”*.

### Cod de înlocuit
```python
MY_PROJECT_NUMBER = "8"
MY_NAME = "Customer Personality Analysis "            # ex. "online_shoppers"
TARGET_C = "Response"             # daca clientul a acceptat oferta din ultima campanie
TASK_C = "classification"               # "classification"

TARGET_R = ""
TASK_R = "regression"               # "regression"
```

### Cod corect
```python
# TODO:
MY_PROJECT_NUMBER = "8"
MY_NAME = "Customer Personality Analysis"   # fara spatiu la final: numele apare in caile fisierelor
TARGET_C = "Response"                       # 1 = clientul a acceptat oferta din ultima campanie, 0 = nu
TASK_C = "classification"

TARGET_R = "MntTotal"                       # suma cheltuita in 2 ani pe toate produsele (o construim la curatare)
TASK_R = "regression"
MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

MY_DIR = f"{DATA_DIR}/{MY_NAME}"
os.makedirs(MY_DIR, exist_ok=True)
print(MY_PROJECT_NUMBER, MY_NAME, TARGET_C, TARGET_R, MY_DIR)
```

`MNT_COLS` stă aici pentru că lista apare de 5 ori în notebook. Definită o singură dată, nu riști să o scrii diferit într-un loc.

### Celula de text (o creezi sub celula de cod)
```markdown
**Proiect 8 — Customer Personality Analysis**
Set de date: [Kaggle — imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), fișierul `marketing_campaign.csv` (2240 clienți × 29 coloane, separator TAB).
Povestea: un lanț de magazine pregătește o nouă campanie de oferte și vrea să știe ce clienți să contacteze și cât valorează fiecare client.

---
- **Clasificare:** prezic dacă un client acceptă oferta campaniei (`Response`) pentru fiecare client din baza de date, ca să trimitem oferta doar celor care au șanse reale să o accepte.
- **Regresie:** prezic suma cheltuită de un client în ultimii 2 ani (`MntTotal` = suma coloanelor `Mnt*`), ca să știm cât valorează fiecare client.
```

---

## TODO 1 — Încărcarea datelor (celula 9) ⚠️

### Ce e greșit
1. TODO-ul cere `f"{MY_DIR}/raw.csv"`. Tu salvezi în `f"{DATA_DIR}/customer/customer.csv"`. Celula de verificare de la final o confirmă: în `MY_DIR` există doar `clean.csv`, **nu și `raw.csv`**.
2. TODO-ul cere un DataFrame numit **`df`**. Tu îl numești `customer`, iar în celula următoare ai nevoie de `if 'df' not in locals(): df = customer.copy()`, un ocoliș care ascunde problema.
3. A rămas comentariul `# EXAMPLE (Penguins)`, care e înșelător într-o celulă TODO.
4. Nu afișezi coloana țintă (pasul 4 din TODO).

### Cod de înlocuit
```python
# EXAMPLE (Penguins)

import kagglehub

CUSTOMER_CSV = f"{DATA_DIR}/customer/customer.csv"

if os.path.exists(CUSTOMER_CSV):
    customer = pd.read_csv(CUSTOMER_CSV)
    ...
print(customer.shape)
customer.head(3)
```

### Cod corect
```python
# TODO: incarcarea setului de date
import kagglehub

RAW_CSV = f"{MY_DIR}/raw.csv"
if os.path.exists(RAW_CSV):                                    # exista deja in Drive -> il citim de acolo
    df = pd.read_csv(RAW_CSV)
    print("Loaded from Drive")
else:                                                          # nu exista -> il descarcam o singura data si il salvam
    path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")   # intoarce un FOLDER
    df = pd.read_csv(f"{path}/marketing_campaign.csv", sep="\t")                     # fisierul original e separat prin TAB
    df.to_csv(RAW_CSV, index=False)                                                  # copia din Drive e separata prin virgula
    print("Downloaded and saved to Drive")

print(df.shape)
print(f"target {TARGET_C}:", sorted(df[TARGET_C].unique()))
df.head(3)
```

**De ce `sep="\t"` doar la descărcare:** fișierul de pe Kaggle e separat prin TAB, dar `to_csv` scrie cu virgulă. La a doua rulare citești copia ta, deci nu mai e nevoie de `sep`.

---

## TODO 2 — Structura tabelului (celula 16) ⚠️

### Ce e greșit
1. `if 'df' not in locals() and 'customer' in locals(): df = customer.copy()` este un ocoliș pentru problema de la TODO 1. Dacă `df` există deja dintr-o rulare anterioară (de exemplu cu `MntTotal` adăugat), **nu se mai reîncarcă**. Exact asta s-a întâmplat la tine: lista `numeric` din output conține `MntTotal`, o coloană care **nu există** în setul original.
2. `df.sample(5, random_state=SEED)` nu e ultima linie din celulă, deci **nu se afișează**. Ultima linie este `print(df["Dt_Customer"].dtype)`. Pentru tabele la mijlocul celulei folosește `display(...)`.
3. Bucla `for c in cat_cols` afișează și cele 663 de date distincte din `Dt_Customer`, adică zgomot. Afișează valorile unice doar pentru coloanele cu puține categorii.
4. Lipsește verificarea care contează cel mai mult la setul tău: **duplicatele fără `ID`**. Cu `ID` inclus, fiecare rând e unic (0 duplicate). Fără `ID` apar 182 de rânduri identice. `ID` ascunde duplicatele.

### Cod de înlocuit
Toată celula, de la `if 'df' not in locals() ...` până la `print(df["Dt_Customer"].dtype)`.

### Cod corect
```python
# TODO: Analiza structurii setului de date
print("shape:", df.shape)
df.info()

# Separarea coloanelor
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
print("\nnumeric:", num_cols)
print("text / categorical:", cat_cols)

# Cardinalitate, duplicate si coloane speciale
print("\ndistinct values per column:\n", df.nunique())
print("\nduplicated rows:", df.duplicated().sum())
print("duplicated rows without ID:", df.drop(columns=["ID"]).duplicated().sum())   # ID face fiecare rand unic si ascunde duplicatele
print("constant columns:", [c for c in df.columns if df[c].nunique(dropna=False) == 1])
print("ID-like columns:", [c for c in df.columns if df[c].nunique() == len(df)])

for c in cat_cols:
    if df[c].nunique() <= 20:                                  # Dt_Customer are 663 de valori: nu le afisam pe toate
        print(f"\n{c}:", df[c].value_counts().to_dict())

# valori imposibile / coduri ascunse in coloanele numerice
print("\nsmallest Year_Birth:", df["Year_Birth"].nsmallest(4).tolist())
print("largest Income:", df["Income"].nlargest(3).tolist())
print("Dt_Customer dtype:", df["Dt_Customer"].dtype)

display(df.sample(5, random_state=SEED))
```

Am folosit `value_counts()` în loc de `unique()` pentru că arată și **câte** rânduri are fiecare categorie. Așa vezi imediat că `Absurd`, `YOLO` și `Alone` au câte 2–3 rânduri.

---

## TODO 2 — Observații structură (celula 17) ❌

### Ce e greșit în răspunsul tău
- *„Year_Birth este un cod ascuns”*: **nu**. Un cod ascuns este o valoare care **înseamnă „lipsă”**, scrisă ca valoare normală („?”, -1, 0, 999). 1893, 1899 și 1900 sunt **valori imposibile** (erori de introducere). Le tratăm la fel (devin `NaN`), dar nu e același concept.
- Ai ratat **codul ascuns real**: `Income = 666666`. E o valoare de umplere tipică („6666…”), nu un venit real. Următorul venit ca mărime este 162397.
- Ai ratat categoriile fără sens din `Marital_Status`: `Absurd`, `YOLO` (nu sunt stări civile) și `Alone` (= `Single`).
- La coloane de eliminat lipsește **`ID`** (identificator).
- Lipsesc duplicatele: 182 de rânduri identice după ce scoți `ID`.

### Răspuns corect (înlocuiește tot textul din celulă)
```markdown
**Observații:**
- **Coloane de eliminat:** `ID` (identificator: o valoare diferită pe fiecare rând, nu descrie clientul); `Z_CostContact` și `Z_Revenue` (constante: aceeași valoare, 3 și 11, pe toate rândurile, deci nu aduc informație).
- **Tip greșit:** `Dt_Customer` (data înscrierii clientului) este citită ca text (`object`). O transformăm în dată, apoi într-un număr: vechimea clientului în zile.
- **Coduri ascunse și valori imposibile:**
  - `Income = 666666`: cod de umplere, nu un venit real (următorul venit ca mărime este 162397) → `NaN`;
  - `Year_Birth` = 1893, 1899, 1900: imposibil (clienți de peste 110 ani în 2014) → `NaN`;
  - `Marital_Status`: `Absurd` și `YOLO` nu sunt stări civile, iar `Alone` înseamnă `Single`.
- **Duplicate:** 0 cu `ID`, dar **182 fără `ID`**: același client înregistrat de două ori, cu ID-uri diferite.
- **Categorii de verificat:** `Education = "2n Cycle"` înseamnă ciclul 2 Bologna, adică **master**, deci nu e un nivel între `Basic` și `Graduation`.
```

---

## TODO 2 — Celula AI (celula 19) ❌

Celula este **copiată de la alt set de date**: *„setul UCI 544 Obesity Levels… 2.111 instanțe și 17 coloane… 24 de rânduri duplicate”*. Nimic din ea nu e adevărat pentru setul tău (2240 × 29, 0 duplicate cu `ID`). Un profesor vede asta imediat și pune la îndoială tot laboratorul.

Ori o ștergi (dacă nu ai folosit AI la acest pas), ori o completezi cinstit, de exemplu:
```markdown
**Dacă ați folosit un tool AI, completați această celulă**:
- **Promptul folosit:** „Am un DataFrame pandas `df` cu setul Customer Personality Analysis (2240 × 29; coloane numerice + `Education`, `Marital_Status`, `Dt_Customer` ca text). Vreau să găsesc coloanele constante, coloanele de tip identificator, rândurile duplicate și valorile imposibile. Doar pandas; nu modifica datele, doar afișează. Explică fiecare linie.”
- **Codul primit:** `select_dtypes`, `nunique`, `duplicated`, `value_counts` și sugestia de a verifica duplicatele fără `ID`.
- **Rezultat:** `ID` identificator; `Z_CostContact`, `Z_Revenue` constante; 182 de duplicate fără `ID`; `Income = 666666` și `Year_Birth < 1901` imposibile; `Dt_Customer` e text.
- **Cum am verificat:** am rulat celula și am comparat `df.shape` cu pagina Kaggle (2240 de rânduri); am verificat manual cu `df["Income"].nlargest(3)`.
```

---

## TODO 3 — Întrebarea Penguins (celula 26) ❌

Celula a rămas „Răspundeți în această căsuță”. Răspuns:
```markdown
- **Atributul care separă cel mai bine speciile:** `flipper_length_mm` separă Gentoo (media 217 mm) de celelalte (190 și 196 mm); `bill_length_mm` separă Adelie (38,8 mm) de Chinstrap și Gentoo (aprox. 48 mm); `bill_depth_mm` separă din nou Gentoo (15 mm față de 18,3 mm). Niciun atribut nu separă singur toate cele trei specii, dar două luate împreună (lungimea și adâncimea ciocului) o fac aproape perfect (vezi scatter-ul).
- **Clasele sunt echilibrate?** Moderat: 44% Adelie, 36% Gentoo, 20% Chinstrap; raportul de dezechilibru este 2,24.
- **Cea mai corelată pereche:** `flipper_length_mm` – `body_mass_g` (Pearson 0,87): pinguinii cu aripi mai lungi sunt mai grei. Cele două coloane aduc în mare parte aceeași informație (redundanță).
```

---

## TODO 3 — Statistici descriptive (celula 27) ⚠️

### Ce e greșit
1. **Modifici `df`-ul brut:** `df["MntTotal"] = df[mnt_cols].sum(axis=1)`. De aici vine `MntTotal` în outputul de la TODO 2 (rulat a doua oară) și `BEFORE: (2240, 30)` în loc de 29 la curățare. Regula din laborator: explorarea **nu modifică datele**. Toate transformările se fac la curățare, pe o copie.
2. Redefinești ținta ca `target_c = "Response"` / `target_r = "MntTotal"`, deși `TARGET_C` există deja. Două nume pentru același lucru înseamnă că le poți schimba pe una și o uiți pe cealaltă.
3. `ID` și coloanele constante intră în `describe()` și `numeric_summary`. Statisticile pentru `ID` (media unui identificator) nu au sens. Le scoți o singură dată, la început.
4. Lipsește o verificare utilă pentru TODO-ul următor: **diferă rata de răspuns între clienții cu și fără `Income`?** (Da: 4% față de 15%, vezi mai jos.)

Restul (funcția `numeric_summary`, excluderea coloanelor binare de la skew, excluderea `Mnt*` din corelația cu `MntTotal`) e **corect și bine gândit**.

### Cod de înlocuit
Toată celula. Liniile problematice sunt:
```python
target_c = "Response"
...
mnt_cols = ["MntWines", ...]
if "MntTotal" not in df.columns:
    df["MntTotal"] = df[mnt_cols].sum(axis=1)
target_r = "MntTotal"
```

### Cod corect
```python
# TODO: Statistici descriptive
pd.set_option("display.max_columns", None)                 # afisam toate coloanele tabelelor (fara "...")

# lucram pe o copie: fara ID / coloane constante (statisticile lor nu au sens), plus tinta de regresie
# df ramane neschimbat - transformarile reale se fac la capitolul 5 (curatare)
df_stats = df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"]).assign(**{TARGET_R: df[MNT_COLS].sum(axis=1)})

# Statistici descriptive generale
display(df_stats.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(2))
display(df_stats.describe(exclude="number"))

# Tabel cu metrici per coloana
def numeric_summary(data):
    rows = {}
    for c in data.select_dtypes(include="number").columns:
        s = data[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()     # candidati la outlier (IQR)
        rows[c] = {"missing %": 100 * s.isna().mean(), "mean": s.mean(), "median": s.median(),
                   "std": s.std(), "CV": s.std() / s.mean() if s.mean() != 0 else np.nan,
                   "min": s.min(), "max": s.max(), "IQR": iqr, "IQR outliers": outside,
                   "skew": s.skew(), "kurtosis": s.kurt()}
    return pd.DataFrame(rows).T.round(2)

summary = numeric_summary(df_stats)
display(summary)

# asimetria, fara coloanele binare 0/1 (acolo skew masoara doar cat de rara e valoarea 1)
binary = [c for c in summary.index if set(df_stats[c].dropna().unique()) <= {0, 1}]
skew = summary.drop(index=binary)["skew"]
print("strongly skewed (|skew| > 1):\n", skew[skew.abs() > 1].sort_values(ascending=False))
print("\nmoderately skewed (0.5 - 1):\n", skew[skew.abs().between(0.5, 1)].sort_values(ascending=False))

# Valori lipsa
missing = 100 * df_stats.isna().mean()
print("\nmissing values (%):\n", missing[missing > 0].round(2))
print("missing Income (rows):", df_stats["Income"].isna().sum())

# Lipsa venitului spune ceva despre tinta? (decide daca merita add_indicator in L03)
no_income = df_stats["Income"].isna()
print(f"response rate: without Income = {df_stats.loc[no_income, TARGET_C].mean():.3f} "
      f"| with Income = {df_stats.loc[~no_income, TARGET_C].mean():.3f}")

# Distributia tintei de clasificare
counts = df_stats[TARGET_C].value_counts()
print("\ntarget distribution:\n", df_stats[TARGET_C].value_counts(normalize=True).round(3))
print("imbalance ratio (largest / smallest class):", round(counts.max() / counts.min(), 2))

# Media si mediana fiecarei coloane numerice, pe clase
num_feats = [c for c in df_stats.select_dtypes(include="number").columns if c != TARGET_C]
print("\nmean / median of each numeric column per class:")
display(df_stats.groupby(TARGET_C)[num_feats].agg(["mean", "median"]).round(1).T)   # .T: un rand pe atribut, mai usor de citit

# Tabele de contingenta: rata de raspuns pe categorii
for c in ["Education", "Marital_Status"]:
    print(f"{c} x {TARGET_C} (share of responders):")
    display(pd.crosstab(df_stats[c], df_stats[TARGET_C], normalize="index").round(3))

# Corelatiile Pearson si Spearman
corr_cols = df_stats.select_dtypes(include="number").columns
pearson = df_stats[corr_cols].corr(method="pearson")
spearman = df_stats[corr_cols].corr(method="spearman")
display(pearson.round(2))

pairs = pearson.where(np.triu(np.ones(pearson.shape, dtype=bool), k=1)).stack()     # fiecare pereche o singura data
print("strongest pairs (Pearson):\n", pairs.abs().sort_values(ascending=False).head(5).round(2))

# corelatia cu tinta de clasificare
print(f"\ncorrelation with {TARGET_C} (Spearman):\n",
      spearman[TARGET_C].drop(TARGET_C).sort_values(key=abs, ascending=False).head(10).round(2))

# corelatia cu tinta de regresie, fara componentele Mnt* (MntTotal e chiar suma lor -> corelatie trivial mare)
print(f"\ncorrelation with {TARGET_R} (Spearman):\n",
      spearman[TARGET_R].drop([TARGET_R] + MNT_COLS).sort_values(key=abs, ascending=False).round(2))
```

**Ce am schimbat și de ce:**
- `df_stats = df.drop(...).assign(...)` creează o **copie**; `df` rămâne exact cum l-ai încărcat. `assign(**{TARGET_R: ...})` adaugă coloana cu numele din constantă, fără să scrii `"MntTotal"` de mână.
- `.T` la tabelul pe clase: cu 26 de atribute × 2 statistici, tabelul lat nu se poate citi. Transpus, ai un rând pe atribut și două coloane pe clasă.
- Crosstab-urile cu `normalize="index"` dau **rata de răspuns** pe categorie (de ex. PhD 20,8% față de Basic 3,7%). Numerele absolute, ca în exemplu, nu se pot compara între categorii de mărimi foarte diferite.
- Am adăugat corelația cu `Response`, pentru că întrebarea „ce atribute par legate de țintă?” se referă în primul rând la ținta de clasificare.

---

## TODO 3 — Observații statistici (celula 28) ❌

### Ce e greșit în răspunsul tău
- *„Ce coloane sunt asimetrice? — coloana skew”*: nu ai răspuns la întrebare, ai indicat doar unde e răspunsul.
- *„O valoare este lipsă la Income”*: **greșit**. Lipsesc **24 de valori** (1,07% din 2240). 1,07% este procentul, nu o singură valoare.
- *„Income… deoarece este pozitiv”*: semnul nu arată cât de puternică e legătura. Contează **mărimea** corelației, iar semnul arată doar direcția. Mai mult, `Income` (0,85) nu este nici cea mai mare: `NumCatalogPurchases` are 0,89.
- Nu ai răspuns la *„Ținta este echilibrată?”* și *„leakage?”*.

### Răspuns corect
```markdown
**Observații:**
- **Coloane asimetrice:** puternic asimetrice spre dreapta (skew > 1): `Income` (6,76, din cauza valorii 666666), `NumDealsPurchases` (2,42), toate coloanele `Mnt*` (1,2–2,1), `NumCatalogPurchases` (1,88), `NumWebPurchases` (1,38). Moderat: `MntTotal` (0,86), `NumStorePurchases` (0,70). La toate media este mult peste mediană (de ex. `MntWines`: media 304, mediana 174), deci pentru completarea lipsurilor folosim **mediana**.
- **Valori lipsă:** doar `Income`: 24 de valori (1,07%). Sunt puține, deci completăm cu mediana, după împărțirea train/test. Rata de răspuns la clienții fără venit este 4% față de 15% la ceilalți, dar e calculată pe doar 24 de rânduri (1 singur client care a răspuns), deci nu e o dovadă solidă. O testăm cu `add_indicator` în L03.
- **Ținta este echilibrată?** **Nu**: 85,1% nu au răspuns (0), 14,9% au răspuns (1); raportul de dezechilibru este 5,71. Un model care spune mereu „0” are acuratețe 0,85, deci folosim acuratețea echilibrată, F1 și ROC AUC.
- **Atribute legate de țintă (`Response`):** clienții care au răspuns au `Recency` mai mic (au cumpărat recent: mediana 30 față de 52 de zile), cheltuie mai mult (`MntWines`: 448 față de 152, `MntMeatProducts`: 174 față de 56), au venit mai mare (64k față de 50k), au acceptat mai des campaniile anterioare (`AcceptedCmp*`) și au mai puțini copii (`Kidhome`, `Teenhome`). Rata de răspuns crește cu educația (Basic 3,7% → PhD 20,8%) și e mai mare la Single/Divorced/Widow decât la Married/Together.
- **Atribute legate de `MntTotal`:** `NumCatalogPurchases` (0,89), `Income` (0,85), `NumStorePurchases` (0,80), `NumWebPurchases` (0,73) pozitiv; `Kidhome` (−0,62) și `NumWebVisitsMonth` (−0,48) negativ.
- **Leakage:** la regresie, coloanele `Mnt*` **sunt** ținta (`MntTotal` e suma lor) → le scoatem din X. `Num*Purchases` sunt numărate în aceeași perioadă de 2 ani ca `Mnt*`: le știm doar după ce clientul a cheltuit deja. Le discutăm la TODO 6. La clasificare nu am găsit o coloană suspectă: nicio corelație cu `Response` nu e apropiată de ±1 (cea mai mare este `AcceptedCmp5`, ≈ 0,33).
```

---

## TODO 3 — Celula AI (celula 30) ⚠️

E incompletă („Codul primit” și „Cum am verificat” sunt goale). Completează-o cinstit sau șterge-o. De exemplu: *„Cum am verificat: am comparat skew-ul cu poziția mediei față de mediană din `describe()`: unde media ≫ mediana, skew > 1.”*

---

## TODO 4 — Grafice (celula 35) ✅ / ⚠️

Graficele sunt **bune**: fiecare răspunde la o întrebare, au etichete, au distribuția țintei (cu procente) și o caracteristică față de țintă, iar figura se salvează înainte de `show`. Excluderea `Income ≥ 200000` doar pentru afișare e o decizie bună.

### Ce e de corectat
1. Același `df["MntTotal"] = ...`, care **modifică `df`**. Folosește `df_stats` din TODO 3.
2. `target_c = "Response"` duplică `TARGET_C`.
3. Comentariile de la final (*„Dacă primești eroare tick_labels… înlocuiește atunci cu labels=”*, *„Variabile necesare: FIGURES_DIR…”*) sunt **notițe primite de la AI, adresate ție**, nu comentarii de cod. Șterge-le. Dacă vrei să păstrezi explicațiile, mută-le într-o celulă de text.

### Cod de înlocuit
```python
target_c = "Response"                                # tinta de clasificare
classes = sorted(df[target_c].unique())              # [0, 1]

# ne asiguram ca tinta de regresie exista
mnt_cols = [...]
if "MntTotal" not in df.columns:
    df["MntTotal"] = df[mnt_cols].sum(axis=1)
```
plus toate aparițiile `df[...]` → `df_stats[...]`, `target_c` → `TARGET_C`, `"MntTotal"` → `TARGET_R`, și cele 4 comentarii de la final.

### Cod corect
```python
# TODO: minim 4 plot-uri, fiecare raspunde la o intrebare (titlul)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
classes = sorted(df_stats[TARGET_C].unique())                # [0, 1]
labels = ["No (0)", "Yes (1)"]
colors = ["indianred", "seagreen"]                           # aceeasi culoare pentru aceeasi clasa in toate graficele
no_outlier = df_stats[df_stats["Income"] < 200_000]          # fara 666666, DOAR pentru afisare

# 1) Cum este distribuit venitul clientilor?
axes[0, 0].hist(no_outlier["Income"], bins=40, color="steelblue")
axes[0, 0].set(title="How is customer income distributed?", xlabel="Income (per year)", ylabel="customers")

# 2) Cheltuie mai mult clientii care au raspuns la campanie?
groups = [df_stats.loc[df_stats[TARGET_C] == k, TARGET_R] for k in classes]
axes[0, 1].boxplot(groups, tick_labels=labels)
axes[0, 1].set(title="Do responders spend more in total?", xlabel=TARGET_C, ylabel=f"{TARGET_R} (2 years)")

# 3) Este tinta echilibrata?
vc = df_stats[TARGET_C].value_counts().sort_index()
axes[0, 2].bar(labels, vc.values, color=colors)
for i, v in enumerate(vc.values):
    axes[0, 2].text(i, v, f"{100 * v / vc.sum():.1f}%", ha="center", va="bottom")
axes[0, 2].set(title=f"Is the target ({TARGET_C}) balanced?", ylabel="customers")

# 4) Explica venitul cat cheltuie un client? (si unde sunt cei care au raspuns)
for k, color, label in zip(classes, colors, labels):
    part = no_outlier[no_outlier[TARGET_C] == k]
    axes[1, 0].scatter(part["Income"], part[TARGET_R], s=10, alpha=0.5, color=color, label=label)
axes[1, 0].set(title="Does income drive total spending?", xlabel="Income (per year)", ylabel=f"{TARGET_R} (2 years)")
axes[1, 0].legend(title=TARGET_C)

# 5) Ce caracteristici sunt corelate? (doar 12 coloane, ca valorile din celule sa se poata citi)
heat_cols = ["Income", "Year_Birth", "Kidhome", "Teenhome", "Recency", "NumDealsPurchases", "NumWebPurchases",
             "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", TARGET_R, TARGET_C]
corr = df_stats[heat_cols].corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(heat_cols)), heat_cols, rotation=45, ha="right", fontsize=8)
axes[1, 1].set_yticks(range(len(heat_cols)), heat_cols, fontsize=8)
for i in range(len(heat_cols)):
    for j in range(len(heat_cols)):
        axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
axes[1, 1].set_title("Which features are correlated? (Pearson)")
fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

# 6) Difera rata de raspuns in functie de educatie? (proportii: grupurile au marimi foarte diferite)
pd.crosstab(df_stats["Education"], df_stats[TARGET_C], normalize="index").plot(
    kind="bar", stacked=True, ax=axes[1, 2], rot=0, color=colors)
axes[1, 2].set(title="Does response rate differ by education?", ylabel="share of customers")
axes[1, 2].legend(labels, title=TARGET_C)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/L02_{MY_NAME}_explore.png", dpi=150, bbox_inches="tight")   # salvam INAINTE de show
plt.show()
```

---

## TODO 5 — Lista de decizii (celula 43) ❌

Celula e necompletată. TODO-ul cere deciziile **înainte** de cod:
```markdown
**Lista de decizii:**
1. Elimin `ID`: identificator, nu descrie clientul.
2. Elimin `Z_CostContact` și `Z_Revenue`: constante (3 și 11 pe toate rândurile).
3. Elimin duplicatele **după** ce scot `ID`: 182 de rânduri identice = același client înregistrat de două ori. Dacă le păstrez, același client poate ajunge și în train, și în test, iar scorul iese prea optimist.
4. `Income = 666666` → `NaN`: cod de umplere, nu un venit real. Păstrez rândul (are țintă validă) și completez venitul mai târziu, din train.
5. `Year_Birth < 1920` (1893, 1899, 1900) → `NaN`: valori imposibile; păstrez rândurile din același motiv.
6. `Marital_Status`: `Alone` → `Single` (același înțeles); `Absurd`, `YOLO` → `Other` (răspunsuri fără sens, 4 rânduri).
7. `Education`: `2n Cycle` → `Master`: „2n Cycle” este ciclul 2 Bologna, adică master. Altfel aș avea două categorii pentru același nivel, iar ordinea (L03) ar ieși greșită.
8. `Dt_Customer`: text → dată → `Customer_Days` = câte zile au trecut de la înscriere până la 31.12.2014 (o dată fixă, după ultima înscriere din set: 29.06.2014).
9. Construiesc ținta de regresie `MntTotal` = suma celor 6 coloane `Mnt*`.
10. **Nu** completez lipsurile din `Income` / `Year_Birth` acum: completarea învață o mediană, deci se face doar pe train (în pipeline).
```

---

## TODO 5 — Curățarea (celula 44) ⚠️

### Ce e greșit
1. **Ștergi rândurile cu `Income > 600_000` și `Year_Birth < 1920`.** Laboratorul spune: *„valori lipsă codificate (…, 0 imposibil) → le transformăm în `NaN`”*. `666666` este exact un cod pentru „lipsă”. Ștergând rândul, pierzi și `Response` și celelalte 25 de coloane, care sunt valide. Corect: `NaN` și completare din train.
2. **`dt.max()` ca dată de referință** depinde de date. Pe un client nou (de exemplu înscris în 2015), `dt.max()` se schimbă, deci și semnificația coloanei `Customer_Days`. O dată fixă (`2014-12-31`) dă aceeași valoare indiferent ce rânduri ai.
3. **`pd.to_datetime(..., dayfirst=True)`** ghicește formatul. `format="%d-%m-%Y"` e explicit și dă eroare dacă apare o dată cu alt format, în loc să o interpreteze greșit fără să te anunțe.
4. **`2n Cycle`** a rămas ca o categorie separată (vezi decizia 7).
5. `print("BEFORE:", df.shape)` afișa `(2240, 30)`, pentru că `df` fusese deja modificat (`MntTotal`). După corecturile de mai sus va afișa `(2240, 29)`.
6. Comentariile de la final (*„Dacă în notebook ai luat alte decizii la pasul anterior…”*) sunt din nou notițe de la AI → le ștergi.

### Cod de înlocuit
```python
# 3) Valori imposibile: an nasterii < 1920 (clienti de peste 90 de ani in 2014) si venitul extrem 666666
impossible = (df_clean["Year_Birth"] < 1920) | (df_clean["Income"] > 600_000)
print("rows with impossible values:", impossible.sum())
df_clean = df_clean[~impossible]
...
dt = pd.to_datetime(df_clean["Dt_Customer"], dayfirst=True)
df_clean["Customer_Days"] = (dt.max() - dt).dt.days
```
plus comentariile de la final.

### Cod corect
```python
# TODO: aplicam deciziile pe o copie a df
print("BEFORE:", df.shape)
display(df.head(3))

df_clean = df.copy()

# 1-2) identificatorul si coloanele constante nu aduc informatie
df_clean = df_clean.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])

# 3) duplicate: fara ID apar clienti inregistrati de doua ori -> pastram o singura copie
print("duplicated rows (without ID):", df_clean.duplicated().sum())
df_clean = df_clean.drop_duplicates()

# 4-5) cod ascuns si valori imposibile -> NaN (randurile raman: tinta si restul coloanelor sunt valide)
df_clean.loc[df_clean["Income"] == 666666, "Income"] = np.nan
df_clean.loc[df_clean["Year_Birth"] < 1920, "Year_Birth"] = np.nan

# 6-7) categorii cu acelasi inteles / fara sens
df_clean["Marital_Status"] = df_clean["Marital_Status"].replace({"Alone": "Single", "Absurd": "Other", "YOLO": "Other"})
df_clean["Education"] = df_clean["Education"].replace({"2n Cycle": "Master"})       # ciclul 2 Bologna = master
print("Marital_Status:", df_clean["Marital_Status"].value_counts().to_dict())
print("Education:", df_clean["Education"].value_counts().to_dict())

# 8) data inscrierii -> vechimea clientului in zile, fata de o data FIXA (ultima inscriere din set: 2014-06-29)
REFERENCE_DATE = pd.Timestamp("2014-12-31")
signup = pd.to_datetime(df_clean["Dt_Customer"], format="%d-%m-%Y")
df_clean["Customer_Days"] = (REFERENCE_DATE - signup).dt.days
df_clean = df_clean.drop(columns=["Dt_Customer"])

# 9) tinta de regresie: suma cheltuita pe toate categoriile de produse
df_clean[TARGET_R] = df_clean[MNT_COLS].sum(axis=1)

# lipsurile raman NaN: le completam mai tarziu, invatand DOAR din setul de antrenament
print("\nmissing after cleaning:\n", df_clean.isna().sum()[df_clean.isna().sum() > 0])

df_clean.to_csv(f"{MY_DIR}/clean.csv", index=False)
print("\nAFTER:", df_clean.shape)
display(df_clean.head(3))
```

**Rezultat așteptat:** `BEFORE: (2240, 29)` → `AFTER: (2058, 27)`; lipsuri: `Income` 25 (24 + valoarea 666666), `Year_Birth` 3.

**De ce am numerotat comentariile 1–9:** corespund punctelor din lista de decizii. Cine citește codul găsește imediat motivul fiecărei linii.

---

## TODO 6 — Întrebările Penguins (celulele 53 și 56) ❌

Ambele au rămas necompletate.

**Celula 53** (verifică pe desenul arborelui din notebook-ul tău primele două niveluri):
```markdown
- **Matricele de confuzie:** regresia logistică nu confundă nicio specie (69/69 corecte). Arborele confundă **un singur Chinstrap cu Adelie** (recall Chinstrap 0,93 = 13/14); Gentoo nu este confundat de niciun model.
- **Drumul unui pinguin prin arbore:** prima întrebare este despre lungimea aripii (`flipper_length_mm`): aripile lungi duc aproape numai la Gentoo. Pe ramura cu aripi scurte, a doua întrebare este despre lungimea ciocului (`bill_length_mm`): cioc scurt → Adelie, cioc lung → Chinstrap. Pragurile sunt pe valori scalate (de ex. `num__flipper_length_mm <= 0,4`).
- **Atributele din primele niveluri** sunt exact cele cu cele mai mari diferențe între mediile pe clase din capitolul 3.
```

**Celula 56:**
```markdown
MAE scade de la 693,2 g (baseline) la 232,2 g (regresia liniară), adică cu **461 g** (−67%). R² crește de la −0,01 la 0,876: modelul elimină 87,6% din eroarea pătratică a baseline-ului.
```

---

## TODO 6 — Primele modele (celula 57) ⚠️

### Ce e greșit
1. **Blocul „Regresia logistică în detaliu” apare de două ori**, identic: antrenezi același model de două ori și afișezi de două ori același raport.
2. **`results` și `results_r` nu se afișează**: nu sunt ultima linie din celulă. Tabelul principal, cerut explicit, lipsește din output. Se văd doar diferențele față de baseline.
3. **Suprascrii variabilele exemplului** (`X_tr`, `X_te`, `preprocess`, `models`, `dt`, `logistic`). Exemplul spune clar: *„tr-train, te-test - scris ca să nu interfereze cu codul vostru”*. Folosește `X_train`, `X_test`, ca în L03. Bonus: `dt` era deja numele datelor calendaristice din curățare.
4. **`MntTotal` rămâne printre atributele de clasificare.** Nu e leakage pentru `Response`, dar este **suma exactă** a celor 6 coloane `Mnt*`, care sunt deja în X. O coloană care e combinație liniară a altora nu aduce informație nouă, iar la regresia logistică ponderile se împart arbitrar între ele. Am scos-o.
5. **Leakage la regresie (logică, nu cod):** `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases` sunt numărate **în aceeași perioadă de 2 ani** în care se adună `MntTotal`. Le știi doar **după** ce clientul a cheltuit. Dacă povestea este „cât va cheltui un client” (ca să decidem cui facem oferte), aceste coloane nu sunt disponibile la momentul predicției. Laboratorul definește leakage-ul exact așa: *„o coloană cunoscută abia după momentul predicției”*. Cu ele: R² ≈ 0,85; fără ele: R² ≈ 0,76. Diferența arată cât din scor venea din aceste coloane. Mai jos am pus o variabilă `LEAKY_R`: dacă profesorul consideră că povestea permite să le folosești, o lași goală.
6. Ai doar 2 modele + baseline la clasificare, iar exemplul din enunț cere „cel puțin două”, deci e ok. Am adăugat kNN ca să fie aceleași modele ca la regresie.
7. Comentariile de la final (*„probabil în jur de 0,85–0,90…”*) sunt **predicții ale AI-ului**, nu rezultatele tale → le ștergi. Observațiile reale merg în celula de text.

### Cod de înlocuit
Toată celula 57. O împart în **3 celule**, ca să le poți rula separat și ca outputul să nu fie un singur bloc lung.

### Cod corect — celula 6a: comparația modelelor de clasificare
```python
# TODO: primele modele - clasificare (Response)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay)

df_clean = pd.read_csv(f"{MY_DIR}/clean.csv")                  # pornim de la tabelul CURATAT

# 1. X = atributele, y = tinta; MntTotal iese: e suma exacta a coloanelor Mnt*, deja prezente in X
X = df_clean.drop(columns=[TARGET_C, TARGET_R])
y = df_clean[TARGET_C]
num_feats = X.select_dtypes(include="number").columns.tolist()  # include coloanele 0/1 (AcceptedCmp*, Complain)
cat_feats = X.select_dtypes(exclude="number").columns.tolist()  # Education, Marital_Status

# 2. I - Isolate: 20% test, stratificat (clasa 1 are doar ~15%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 3. C - Construct: preprocesarea, invatata DOAR din train
def make_preprocess(num, cat):
    return ColumnTransformer([
        # mediana: Income e foarte asimetric (skew 6,8)
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
preprocess_c = make_preprocess(num_feats, cat_feats)

# 4. baseline + modele, fiecare intr-un Pipeline
models_c = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=1000),
    "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "decision tree": DecisionTreeClassifier(max_depth=3, random_state=SEED),
}

# 5. T - Test: 2 clase -> precision / recall / F1 pentru clasa 1, ROC AUC din probabilitatea clasei 1
rows = []
for name, model in models_c.items():
    pipe = Pipeline([("prep", preprocess_c), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                 "balanced accuracy": balanced_accuracy_score(y_test, pred),
                 "precision": precision_score(y_test, pred, zero_division=0),   # baseline nu prezice niciun 1
                 "recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC AUC": roc_auc_score(y_test, proba)})
results_c = pd.DataFrame(rows).set_index("model").round(3)
results_c.to_csv(f"{RESULTS_DIR}/L02_{MY_NAME}_first_models.csv")
display(results_c)
print("difference vs baseline:")
display((results_c - results_c.loc["baseline"]).drop(index="baseline").round(3))
```

### Cod corect — celula 6b: regresia logistică și arborele în detaliu
```python
# 6. Regresia logistica si arborele in detaliu: raport pe clase, ROC AUC, matricea de confuzie
from sklearn import tree

def show_details(model, title):
    pipe = Pipeline([("prep", preprocess_c), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]                    # coloana 1 = probabilitatea clasei pipe.classes_[1] = 1
    print(f"=== {title} ===")
    print(classification_report(y_test, pred, digits=3))
    print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
    plt.title(f"{title} - confusion matrix")
    plt.show()
    return pipe

logistic_c = show_details(LogisticRegression(max_iter=1000), "Logistic regression")
tree_c = show_details(DecisionTreeClassifier(max_depth=3, random_state=SEED), "Decision tree (max_depth=3)")

# arborele desenat; pragurile numerice sunt pe valori SCALATE
plt.figure(figsize=(20, 8))
tree.plot_tree(tree_c.named_steps["model"],
               feature_names=list(tree_c.named_steps["prep"].get_feature_names_out()),
               class_names=[str(k) for k in tree_c.classes_],
               filled=True, rounded=True, proportion=True, fontsize=9)
plt.title("Decision tree (max_depth=3)")
plt.show()
```

**De ce o funcție:** codul pentru regresia logistică și cel pentru arbore era identic, doar modelul diferea. Așa ai fost tentat să-l copiezi (și a ieșit de două ori). Cu `show_details` scrii logica o singură dată, iar un model nou e o singură linie.

### Cod corect — celula 6c: regresie (`MntTotal`)
```python
# Regresie: cat cheltuie in total un client in 2 ani? (MntTotal)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# leakage: Mnt* SUNT tinta (MntTotal = suma lor); Num*Purchases sunt numarate in aceeasi perioada de 2 ani,
# deci le stim abia DUPA ce clientul a cheltuit -> le scoatem (LEAKY_R = [] daca povestea permite sa le folosim)
LEAKY_R = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
Xr = df_clean.drop(columns=[TARGET_R] + MNT_COLS + LEAKY_R)
yr = df_clean[TARGET_R]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)   # fara stratify: tinta e numerica
preprocess_r = make_preprocess(Xr.select_dtypes(include="number").columns.tolist(),
                               Xr.select_dtypes(exclude="number").columns.tolist())

models_r = {
    "baseline": DummyRegressor(strategy="mean"),
    "linear regression": LinearRegression(),
    "kNN (k=5)": KNeighborsRegressor(n_neighbors=5),
    "decision tree": DecisionTreeRegressor(max_depth=5, random_state=SEED),
}
rows = []
for name, model in models_r.items():
    pred = Pipeline([("prep", preprocess_r), ("model", model)]).fit(Xr_train, yr_train).predict(Xr_test)
    rows.append({"model": name, "MAE": mean_absolute_error(yr_test, pred),
                 "RMSE": np.sqrt(mean_squared_error(yr_test, pred)), "R2": r2_score(yr_test, pred)})
results_r = pd.DataFrame(rows).set_index("model").round(3)
results_r.to_csv(f"{RESULTS_DIR}/L02_{MY_NAME}_first_models_regression.csv")
display(results_r)
```

**De ce `make_preprocess(num, cat)`:** la clasificare și la regresie preprocesarea e aceeași, doar listele de coloane diferă. Înainte o scriai de două ori.

---

## TODO 6 — Observații modele (celula 58) ❌

Răspuns, cu cifrele obținute de mine după curățarea corectată (înlocuiește-le cu ale tale dacă diferă):
```markdown
**Observații:**
- **Cel mai bun model (clasificare): regresia logistică**: acuratețe echilibrată 0,753, F1 0,647, ROC AUC 0,912. kNN (0,646 / 0,435 / 0,798) și arborele cu `max_depth=3` (0,621 / 0,386 / 0,769) sunt clar mai slabi.
- **Cât depășește baseline-ul:** baseline-ul are acuratețe 0,847 (prezice mereu „0”), dar acuratețe echilibrată 0,5, F1 0 și ROC AUC 0,5. Regresia logistică câștigă doar +0,066 la acuratețe, dar +0,253 la acuratețea echilibrată, +0,647 la F1 și +0,412 la ROC AUC. Acuratețea ascunde aproape tot câștigul, din cauza dezechilibrului.
- **Problema rămasă:** recall-ul pe clasa 1 este doar 0,52: modelul ratează aproape jumătate dintre clienții care ar accepta oferta, pentru că pragul 0,5 favorizează clasa majoritară.
- **Regresie (`MntTotal`, fără `Mnt*` și `Num*Purchases`):** regresia liniară are R² ≈ 0,76 și MAE ≈ 210, față de baseline (MAE ≈ 477, R² ≈ 0). Cu `Num*Purchases` incluse, R² urca la 0,85, dar acele coloane se cunosc abia după ce clientul a cheltuit (leakage).
- **Ce ar putea îmbunătăți:** (1) `class_weight="balanced"` sau SMOTE pentru clasa rară; (2) alegerea pragului de decizie; (3) hiperparametri aleși prin validare încrucișată, nu pe test; (4) atribute noi: vârsta (`2014 − Year_Birth`), numărul total de copii (`Kidhome + Teenhome`), numărul de campanii acceptate anterior (suma `AcceptedCmp*`); (5) la regresie, logaritmul țintei (`MntTotal` e asimetrică).
```

---

## Celulele AI (celulele 11, 37, 46, 60) ⚠️

- **Celula 11**: ok.
- **Celulele 37, 46, 60**: „Rezultat: mai sus / mai sus / gol” și „Cum am verificat: am analizat codul”. Profesorul cere **ce ai verificat concret**. De exemplu: *„am verificat că `MntTotal` din primul rând = 635 + 88 + 546 + 172 + 88 + 88 = 1617”*, *„am verificat că suma procentelor de pe bare este 100%”*.
- Cinstit vorbind: o parte mare din cod are comentarii scrise în stilul unui asistent AI (*„Dacă primești eroare…”*, *„probabil în jur de 0,85–0,90”*). Dacă ai folosit AI pentru aceste celule, declară-l în celula AI corespunzătoare; laboratorul permite asta, dar cere transparență.

---

## TODO 7 — Final (celula 62) ❌

Outputul arată că în `MY_DIR` există doar `clean.csv`. **`raw.csv` lipsește**, pentru că TODO 1 salva în altă parte. După corecturile de mai sus și o rulare completă trebuie să vezi:
```
.../data/Customer Personality Analysis -> ['clean.csv', 'raw.csv']
.../figures -> ['L02_Customer Personality Analysis_explore.png']
.../results -> ['L02_Customer Personality Analysis_first_models.csv', 'L02_Customer Personality Analysis_first_models_regression.csv']
```
(fără spațiu înainte de `_explore` / `_first_models`). Fișierele vechi, cu spațiu în nume, le poți șterge din Drive.
