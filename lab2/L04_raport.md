# Raport L04 — SVM și SVR (T8 Customer Personality Analysis)

Analiză pe fiecare TODO din `L04_SVM_SVR_v2.ipynb`: **verdict**, **ce e greșit și de ce**, **codul de înlocuit** (dacă ai scris ceva) și **codul corect** (copiezi toată celula).

Legendă: ✅ corect · ⚠️ merge, dar are probleme · ❌ greșit sau necompletat

> Cifrele din observații le-am obținut rulând exact codul din acest raport (scikit-learn 1.9) pe `clean.csv` corectat după raportul L02 (2058 de rânduri). Pe Colab pot ieși mici diferențe; copiază în notebook cifrele care îți ies ție.

## Rezumat

| Capitol | TODO | Verdict | Problema principală |
|---|---|---|---|
| 0 | Setul de date | ✅ | — |
| 1 | Cele două probleme | ❌ | **suprascrie `num_c`/`cat_c` din exemplu → exemplul crapă (`KeyError: 'Year_Birth'`)**; leakage `Num*Purchases` la regresie |
| 1 | Funcția de evaluare | ❌ | necompletată |
| 2 | Efectul lui C și ε | ❌ | necompletat |
| 3 | Scalarea | ❌ | necompletată |
| 4 | Căutarea SVM / SVR | ❌ | necompletată |
| 5 | Timpul de antrenare | ❌ | necompletat |
| 6 | Probabilități, prag, reziduuri | ❌ | necompletat |
| 7 | Ce a învățat SVM-ul | ❌ | necompletat |
| 8 | Tabelul de comparație + testul t | ❌ | necompletat |
| 9 | Găsiți greșelile | ❌ | **rezolvat pe alt set de date (HR Attrition), cu alt model, și „corectează” greșeli care nu există în bloc** |
| 10 | Contribuția | ❌ | necompletată |
| — | Temă (fișa) | ❌ | necompletată |

**Notă despre nume:** funcțiile ajutătoare din exemple folosesc `num_c`, `cat_c`, `num_r`, `cat_r`, `cv5_c`, `cv5_r`. În codul tău folosește **alte nume** (`num_cols_c`, `cv_c` etc.), altfel celulele de exemplu care rulează după celula ta primesc listele tale de coloane și crapă, exact ca la tine în celula 10.

---

## TODO 0 — Setul de date (celula 5) ✅

Corect. Valorile sunt consecvente cu L02/L03 (dacă ai scos spațiul din `MY_NAME` și în L02).

---

## TODO 1 — Cele două probleme (celula 8) ❌

### Ce e greșit
1. **Suprascrii variabilele exemplului.** Celula ta definește `num_c`, `cat_c`, `num_r`, `cat_r`. Celula 10 (exemplul Penguins) rulează **după** a ta și construiește `make_pipeline_ex(..., num_c, cat_c)` pe datele Penguins, deci cu coloanele tale (`Year_Birth`...). De aici eroarea `KeyError: 'Year_Birth'` din outputul celulei 10. TODO-ul cere „listele de coloane”, dar nu impune numele; folosește altele.
2. **Ramura `else` reface curățarea** a treia oară (L03, L04), din altă sursă (`data/customer/customer.csv`) și cu deciziile vechi (șterge rândurile cu 666666, `dt.max()`). Folosește `assert`, ca în raportul L03.
3. **Leakage la regresie:** `Num*Purchases` rămân în `Xr` (vezi raportul L02, TODO 6): sunt numărate în aceiași 2 ani ca `MntTotal`. Ai scos corect `Mnt*` și `Response`, dar nu și pe acestea.
4. Restul e bun: `MntTotal` scos de la clasificare, `Response` scos de la regresie (justificat), stratificare doar la clasificare, aceleași fold-uri pentru toate modelele.

### Cod de înlocuit
Toată celula 8, în special:
```python
else:                                                  # nu exista -> refacem rapid curatarea din laboratorul 2
    df = pd.read_csv(f"{DATA_DIR}/customer/customer.csv")
    ...
Xr = df.drop(columns=[TARGET_REG, TARGET_CLF] + mnt_cols)
...
num_c = Xc_train.select_dtypes(include="number").columns.tolist()
cat_c = Xc_train.select_dtypes(exclude="number").columns.tolist()    # Education, Marital_Status
num_r = Xr_train.select_dtypes(include="number").columns.tolist()
cat_r = Xr_train.select_dtypes(exclude="number").columns.tolist()    # Education, Marital_Status
```

### Cod corect
```python
# TODO: cele doua probleme din clean.csv, fiecare cu split-ul si validarea ei
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

CLEAN_CSV = f"{MY_DIR}/clean.csv"
assert os.path.exists(CLEAN_CSV), f"Nu exista {CLEAN_CSV}: rulati intai capitolul 5 din laboratorul 2"
df = pd.read_csv(CLEAN_CSV)

MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
# numarate in ACEEASI perioada de 2 ani ca MntTotal -> le stim abia dupa ce clientul a cheltuit (leakage la regresie)
LEAKY_R = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]

# CLASIFICARE: a acceptat clientul oferta? MntTotal iese: e suma exacta a coloanelor Mnt*, care raman in X
Xc = df.drop(columns=[TARGET_CLF, TARGET_REG])
yc = (df[TARGET_CLF] == POS_CLASS).astype(int)

# REGRESIE: cat cheltuie clientul in 2 ani? Mnt* = tinta, Num*Purchases = leakage,
# Response = rezultatul campaniei (nu il stim pentru un client nou)
Xr = df.drop(columns=[TARGET_REG, TARGET_CLF] + MNT_COLS + LEAKY_R)
yr = df[TARGET_REG]

# 80/20; clasificare stratificata (clasa 1 ~15%); datele nu sunt o serie in timp -> impartire aleatoare
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=SEED)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)

# alte nume decat in exemplu (num_c, cat_c, ...), ca sa nu stricam celulele Penguins
num_cols_c = Xc_train.select_dtypes(include="number").columns.tolist()
cat_cols_c = Xc_train.select_dtypes(exclude="number").columns.tolist()     # Education, Marital_Status
num_cols_r = Xr_train.select_dtypes(include="number").columns.tolist()
cat_cols_r = Xr_train.select_dtypes(exclude="number").columns.tolist()

# aceleasi fold-uri pentru TOATE modelele din laborator
cv_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_r = KFold(n_splits=5, shuffle=True, random_state=SEED)

print("classification:", Xc_train.shape, "| share of class 1:", round(yc_train.mean(), 3))
print("regression:    ", Xr_train.shape, "| target mean:", round(yr_train.mean(), 1), "| median:", yr_train.median())
print("regression features:", num_cols_r + cat_cols_r)
```

### Observații (le adaugi într-o celulă de text sub cod)
```markdown
- **Clasificare (`Response`):** scot ținta și `MntTotal` (suma exactă a coloanelor `Mnt*`, care rămân în X: nu aduce informație nouă).
- **Regresie (`MntTotal`):** scot ținta, cele 6 coloane `Mnt*` (ținta este suma lor), `Num*Purchases` (numărate în aceeași perioadă de 2 ani, deci cunoscute abia după ce clientul a cheltuit) și `Response` (rezultatul campaniei, necunoscut pentru un client nou).
- **Împărțire:** 80/20 cu `SEED`; stratificată la clasificare (clasa 1 are 15%), aleatoare la regresie. Datele nu sunt o serie în timp (`Customer_Days` e o vechime, nu o dată de măsurare), deci nu împart cronologic.
- **Validare:** `StratifiedKFold(5)` la clasificare, `KFold(5)` la regresie, ambele cu `shuffle=True` și aceleași fold-uri pentru toate modelele.
```

---

## TODO 1 — Funcția de evaluare (celula 11) ❌

Celula e goală. TODO-ul cere explicit **funcția voastră**, nu `run_cv` copiat.

### Cod corect
```python
# TODO: functia de evaluare, folosita in tot laboratorul
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression

# prima metrica = metrica principala
METRICS = {
    "classification": {"PR_AUC": "average_precision", "F1": "f1", "recall": "recall", "bal_acc": "balanced_accuracy"},
    "regression":     {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
}

def evaluate(model, X, y, task, cv):
    """CV pe metricile problemei; intoarce (randul de tabel, scorurile pe fold-uri ale metricii principale)."""
    scoring = METRICS[task]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    sign = {name: -1 if scorer.startswith("neg_") else 1 for name, scorer in scoring.items()}   # erorile fara minus
    main = next(iter(scoring))
    folds = sign[main] * res[f"test_{main}"]
    row = {main: folds.mean(), f"{main}_std": folds.std(), f"{main}_train": sign[main] * res[f"train_{main}"].mean()}
    row |= {name: sign[name] * res[f"test_{name}"].mean() for name in scoring if name != main}
    row["fit_time_s"] = res["fit_time"].mean()
    return row, folds

# preprocesarea din L03: mediana + scalare, Education ordinal, Marital_Status one-hot
EDU_ORDER = [["Basic", "Graduation", "Master", "PhD"]]

def make_pipe(model, num_cols, scaler=None):
    """prep + model; scaler=None -> StandardScaler (obligatoriu la SVM), "passthrough" -> fara scalare."""
    num_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler() if scaler is None else scaler)])
    ord_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=EDU_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))])
    nom_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist"))])
    prep = ColumnTransformer([("num", num_steps, num_cols), ("ord", ord_steps, ["Education"]),
                              ("nom", nom_steps, ["Marital_Status"])])
    return Pipeline([("prep", prep), ("model", model)])

# verificare pe modelele simple: acestea sunt scorurile pe care SVM / SVR trebuie sa le bata
row_lr, _ = evaluate(make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), num_cols_c),
                     Xc_train, yc_train, "classification", cv_c)
row_lin, _ = evaluate(make_pipe(LinearRegression(), num_cols_r), Xr_train, yr_train, "regression", cv_r)
print("logistic regression:", pd.Series(row_lr).round(3).to_dict())
print("linear regression:  ", pd.Series(row_lin).round(3).to_dict())
```

**De ce așa:**
- `METRICS` e un dicționar, nu un `if`: alegerea metricilor stă într-un singur loc și se vede dintr-o privire.
- `row |= {...}` (Python 3.9+) adaugă metricile secundare în același dicționar.
- `make_pipe` păstrează deciziile din L03 (Education ordinal). `StandardScaler` implicit: SVM-ul are nevoie ca toate coloanele să fie pe aceeași scară, iar în L03 Standard și Robust au ieșit la egalitate.

### Observații
```markdown
**Metrica principală:**
- **Clasificare → PR-AUC** (`average_precision`). Clasa care ne interesează (clienții care acceptă oferta) are doar 15%. PR-AUC măsoară cât de bine ordonează modelul clienții după probabilitatea de răspuns, adică exact ce folosim la o campanie (contactăm întâi clienții cu scor mare), și nu depinde de prag. Un model care prezice mereu „0” are PR-AUC ≈ 0,15 (proporția clasei 1), nu 0,85 ca acuratețea. Ca metrici secundare urmăresc F1 și recall, ca să prind un model degenerat (F1 = 0).
- **Regresie → MAE**: „greșim în medie cu X unități monetare pe client”, ușor de explicat marketingului. Ținta e asimetrică (câțiva clienți cheltuie peste 2000), iar RMSE ar fi dominat de ei. RMSE îl afișez ca metrică secundară.

**Scoruri de referință (CV pe train):** regresia logistică PR-AUC = 0,601 ± 0,065, F1 = 0,562; regresia liniară MAE = 233, R² = 0,715.
```

---

## TODO 2 — Efectul lui C și ε (celula 22) ❌

Celula e goală. Folosesc `plot_svc_2d` și `plot_svr_1d` din exemplu (sunt permise).

**Alegerea atributelor:** la clasificare `Recency` și `MntWines`. Sunt cele mai diferite între clase în L02 (mediana 30 față de 52 de zile; 448 față de 152), nu au lipsuri și au sens împreună („a cumpărat recent și cheltuie mult”). La regresie `Income`, cel mai corelat atribut cu `MntTotal` care rămâne în X. Pentru un desen lizibil folosesc 400 de rânduri din train: cu 1600 de puncte vectorii suport acoperă tot graficul.

### Cod corect
```python
# TODO: efectul lui C (clasificare, 2 atribute) si al lui epsilon (regresie, 1 atribut)
from sklearn.svm import SVC, SVR
from sklearn.compose import TransformedTargetRegressor

# CLASIFICARE: Recency + MntWines, 400 de randuri din train (stratificat), doar pentru desen
X2, _, y2, _ = train_test_split(Xc_train[["Recency", "MntWines"]], yc_train, train_size=400,
                                stratify=yc_train, random_state=SEED)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, C in zip(axes, [0.01, 1, 100]):
    svm2 = Pipeline([("scale", StandardScaler()),
                     ("model", SVC(kernel="linear", C=C, class_weight="balanced"))]).fit(X2, y2)
    width = 2 / np.linalg.norm(svm2["model"].coef_[0])
    plot_svc_2d(svm2, X2, y2, ax, f"C = {C}: {svm2['model'].n_support_.sum()} support vectors\n"
                                  f"margin width (scaled) = {width:.2f}")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/L04_{MY_NAME}_C_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# REGRESIE: MntTotal in functie de Income, trei latimi ale tubului (tinta scalata -> epsilon in abateri standard)
x1 = Xr_train[["Income"]].dropna().sample(400, random_state=SEED)
y1 = yr_train.loc[x1.index]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, eps in zip(axes, [0.1, 0.5, 1.0]):
    svr1 = TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear", C=1, epsilon=eps))]),
        transformer=StandardScaler()).fit(x1, y1)
    n_sv = len(svr1.regressor_["model"].support_)
    plot_svr_1d(svr1, x1, y1, ax, f"epsilon = {eps} (scaled): {n_sv} of {len(x1)} are support vectors")
axes[0].set_ylabel(TARGET_REG)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/L04_{MY_NAME}_epsilon_effect.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Observații
```markdown
**Observații:**
- **C (clasificare, `Recency` + `MntWines`, 400 de clienți):** la C = 0,01 marginea e largă (3,04 în unități scalate) și 337 din 400 de puncte sunt vectori suport. La C = 1 marginea se îngustează (2,07), iar vectorii suport scad la 287. Între C = 1 și C = 100 nu se mai schimbă nimic: frontiera liniară s-a stabilizat, iar punctele rămase în margine sunt cele care se suprapun între clase și nu pot fi separate de nicio dreaptă.
- **Separabilitate:** **nu**, clasele nu sunt separabile liniar în aceste două atribute. Chiar și cu C = 100, acuratețea pe aceste puncte rămâne ≈ 0,68, iar 72% dintre puncte sunt vectori suport. Clienții care răspund sunt amestecați cu ceilalți; două atribute nu ajung.
- **ε (regresie, `Income` → `MntTotal`):** ε = 0,1 (tub ±62) → 346 din 400 de puncte sunt vectori suport; ε = 0,5 (±308) → 139; ε = 1,0 (±615) → doar 38. Cu cât tubul e mai lat, cu atât mai puține puncte ies din el și contează la antrenare, iar modelul devine mai simplu (mai „plat”).
```

---

## TODO 3 — Scalarea e obligatorie (celula 31) ❌

### Cod corect
```python
# TODO: scalarea la SVM (clasificare) si scalarea tintei la SVR (regresie), cu cifre
rows = []
for name, scaler, cw in [("SVC rbf, no scaling", "passthrough", "balanced"),
                         ("SVC rbf + StandardScaler", None, None),
                         ("SVC rbf + StandardScaler + class_weight", None, "balanced")]:
    row, _ = evaluate(make_pipe(SVC(kernel="rbf", class_weight=cw), num_cols_c, scaler=scaler),
                      Xc_train, yc_train, "classification", cv_c)
    rows.append({"model": name, **row})
display(pd.DataFrame(rows).set_index("model").round(3))

variants_r = {
    "linear regression (reference)": make_pipe(LinearRegression(), num_cols_r),
    "SVR rbf, no scaling":           make_pipe(SVR(kernel="rbf"), num_cols_r, scaler="passthrough"),
    "SVR rbf + scaling X":           make_pipe(SVR(kernel="rbf"), num_cols_r),
    "SVR rbf + scaling X and y":     TransformedTargetRegressor(make_pipe(SVR(kernel="rbf"), num_cols_r),
                                                                transformer=StandardScaler()),
}
rows = []
for name, pipe in variants_r.items():
    row, _ = evaluate(pipe, Xr_train, yr_train, "regression", cv_r)
    rows.append({"model": name, **row})
display(pd.DataFrame(rows).set_index("model").round(3))
```

### Observații
```markdown
**Observații:**
- **Clasificare:** fără scalare, SVC-ul RBF are PR-AUC 0,307, abia peste baseline (0,152). Distanța e dominată de coloanele cu valori mari (`Income` în zeci de mii, `Customer_Days` în sute) și restul atributelor nu mai contează. Cu `StandardScaler`, PR-AUC urcă la 0,567. **Dar** fără `class_weight` modelul e aproape degenerat: F1 0,317, recall 0,216 (prezice aproape mereu 0). PR-AUC nu vede asta, pentru că se uită doar la ordinea scorurilor. Cu `class_weight="balanced"`: F1 0,552, recall 0,752.
- **Regresie:** SVR fără scalarea țintei e **mai slab decât baseline-ul aproape**: cu X scalat, R² = 0,01 (MAE 456). Ținta e în unități monetare (medie ≈ 614, abatere ≈ 600), iar `epsilon = 0,1` și `C = 1` sunt minuscule față de ea: penalizarea pentru o eroare de câteva sute este prea mică, iar SVR-ul rămâne aproape la o constantă. Cu X **și** y scalate, ε = 0,1 înseamnă 0,1 abateri standard (≈ 60), iar SVR-ul ajunge la MAE 194, R² 0,786, mai bine decât regresia liniară (MAE 233).
```

---

## TODO 4 — Căutarea SVM / SVR (celula 41) ❌

### Decizii (le scrii și în notebook, ca text)
- **Kerneluri:** `linear` și `rbf`. `poly` l-am lăsat afară: e mult mai lent, iar `rbf` acoperă deja frontierele curbe. `linear` e referința „simplă”.
- **Scara:** C și γ pe scară logaritmică (efectul lor e multiplicativ). La kernelul liniar C ≤ 10: peste această valoare, libsvm converge foarte greu pe date care nu sunt separabile.
- **ε (SVR):** 0,05 / 0,1 / 0,3, în abateri standard ale țintei (ținta e scalată cu `TransformedTargetRegressor`).
- **`class_weight="balanced"` fixat** la clasificare: clasa 1 are 15%, iar fără el SVC-ul degenerează (TODO 3).
- **`GridSearchCV` pe tot train-ul:** 1646 de rânduri sunt puține, deci nu e nevoie de subeșantion sau de `RandomizedSearchCV`.

### Cod corect
```python
# TODO: cel mai bun SVM (clasificare) si cel mai bun SVR (regresie)
from sklearn.model_selection import GridSearchCV

param_grid_svc = [
    {"model__kernel": ["linear"], "model__C": np.logspace(-2, 1, 4)},                                  # 0.01 ... 10
    {"model__kernel": ["rbf"], "model__C": np.logspace(-1, 3, 5), "model__gamma": np.logspace(-4, 0, 5)},
]
grid_svc = GridSearchCV(make_pipe(SVC(class_weight="balanced"), num_cols_c), param_grid_svc, cv=cv_c,
                        scoring="average_precision", n_jobs=-1, return_train_score=True)
t0 = time.time()
grid_svc.fit(Xc_train, yc_train)                                  # 4 + 25 = 29 combinatii x 5 fold-uri
print(f"SVC search: {time.time() - t0:.0f} s | best: {grid_svc.best_params_} | CV PR-AUC = {grid_svc.best_score_:.3f}")

prefix = "regressor__model__"
param_grid_svr = [
    {f"{prefix}kernel": ["linear"], f"{prefix}C": np.logspace(-2, 1, 4), f"{prefix}epsilon": [0.05, 0.1, 0.3]},
    {f"{prefix}kernel": ["rbf"], f"{prefix}C": np.logspace(-1, 2, 4), f"{prefix}gamma": np.logspace(-3, 0, 4),
     f"{prefix}epsilon": [0.05, 0.1, 0.3]},
]
grid_svr = GridSearchCV(TransformedTargetRegressor(make_pipe(SVR(), num_cols_r), transformer=StandardScaler()),
                        param_grid_svr, cv=cv_r, scoring="neg_mean_absolute_error", n_jobs=-1, return_train_score=True)
t0 = time.time()
grid_svr.fit(Xr_train, yr_train)                                  # 12 + 48 = 60 combinatii x 5 fold-uri
print(f"SVR search: {time.time() - t0:.0f} s | best: {grid_svr.best_params_} | CV MAE = {-grid_svr.best_score_:.1f}")

# cele mai bune 5 combinatii
for grid, title in [(grid_svc, "SVC"), (grid_svr, "SVR")]:
    cols = [c for c in grid.cv_results_ if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_train_score"]
    print(title)
    display(pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[cols].head(5).round(3))

# hartile C x gamma (train vs validare); la SVR fixam epsilon la valoarea celui mai bun model
cg_heatmaps(grid_svc, title="SVC rbf, PR-AUC")
plt.show()
cg_heatmaps(grid_svr, prefix=prefix, error=True, title="SVR rbf, MAE",
            fixed={f"{prefix}epsilon": grid_svr.best_params_.get(f"{prefix}epsilon")})
plt.show()

# modelul ales nu e degenerat?
row_svc, _ = evaluate(grid_svc.best_estimator_, Xc_train, yc_train, "classification", cv_c)
print("best SVC:", pd.Series(row_svc).round(3).to_dict())
```

### Observații
```markdown
**Observații:**
- **Clasificare:** câștigă **RBF** cu C = 100, γ = 0,001 (PR-AUC 0,606 ± 0,063). Al doilea kernel, **liniar** cu C = 0,01, are 0,599 ± 0,083: diferența (0,007) e de 10 ori mai mică decât abaterea standard, deci practic sunt la egalitate. Un γ atât de mic face frontiera RBF aproape liniară, ceea ce confirmă că problema e în mare parte liniară.
- **Zona stabilă (SVC):** pe harta de validare e o **diagonală**: γ = 0,0001 cu C = 10–1000, γ = 0,001 cu C = 1–100 și γ = 0,01 cu C = 0,1–1 dau toate 0,57–0,61. C mai mare cere γ mai mic. **Overfitting** pentru γ ≥ 0,1: pe train 0,79–0,98, pe validare doar 0,44–0,50 (la γ = 1: train ≈ 1, validare 0,28). Și γ = 0,01 cu C = 1000 trece de 0,97 pe train, dar are doar 0,48 pe validare.
- **Regresie:** câștigă **RBF** cu C = 100, γ = 0,01, ε = 0,05 (MAE 199,1 ± 10,9). Următoarea combinație (C = 1, γ = 0,1) are 200,2: aceeași zonă. Cel mai bun kernel liniar e la ≈ 229 (vezi LinearSVR în tabelul din capitolul 8), adică ≈ 30 mai rău, de 3 ori abaterea standard: aici RBF-ul chiar ajută. Diferența train – validare (132 față de 199) arată un overfitting moderat.
- **Modelul ales nu e degenerat:** SVC-ul are F1 0,551, recall 0,776, acuratețe echilibrată 0,795.
```

---

## TODO 5 — Timpul de antrenare (celula 47) ❌

Train-ul are 1646 de rânduri (sub 10.000), deci nu e obligatoriu să compari efectiv alternativa. O fac totuși, pentru că e ieftin și răspunde direct la „ce pierdeți / ce câștigați”.

### Cod corect
```python
# TODO: cat ar dura SVM-ul pe un set de 10 ori mai mare? + o alternativa
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.base import clone

best_svc = grid_svc.best_estimator_
sizes, times = [400, 800, 1200, len(Xc_train)], []
for n in sizes:
    Xs, _, ys, _ = (train_test_split(Xc_train, yc_train, train_size=n, stratify=yc_train, random_state=SEED)
                    if n < len(Xc_train) else (Xc_train, None, yc_train, None))
    t0 = time.time()
    clone(best_svc).fit(Xs, ys)
    times.append(time.time() - t0)

slope = np.polyfit(np.log(sizes), np.log(times), 1)[0]           # timp ~ n^slope
print("fit times (s):", dict(zip(sizes, np.round(times, 3))))
print(f"growth exponent ~ {slope:.2f} -> 10x more rows ~ {times[-1] * 10 ** slope:.1f} s per fit "
      f"(x {29 * 5} fits in the grid search)")

# alternativa: Nystroem (aproximeaza kernelul RBF) + LinearSVC, pe ACELEASI fold-uri
gamma = best_svc["model"].gamma if best_svc["model"].kernel == "rbf" else "scale"
alt = make_pipe(Pipeline([("map", Nystroem(kernel="rbf", gamma=gamma, n_components=300, random_state=SEED)),
                          ("svm", LinearSVC(C=0.1, class_weight="balanced", max_iter=10000))]), num_cols_c)
rows = []
for name, model in [("SVC (best, kernel)", best_svc), ("Nystroem + LinearSVC", alt)]:
    row, _ = evaluate(model, Xc_train, yc_train, "classification", cv_c)
    rows.append({"model": name, **row})
display(pd.DataFrame(rows).set_index("model")[["PR_AUC", "PR_AUC_std", "F1", "fit_time_s"]].round(3))
```

### Observații
```markdown
**Observații:**
- O antrenare a celui mai bun SVC pe tot train-ul (1646 de rânduri) durează ≈ 0,06 s. Pe subeșantioane, timpul crește cu un exponent ≈ 0,85, deci la acest volum încă aproape liniar (costul fix domină). Teoretic, crește între n² și n³. Estimarea pentru 10× mai multe rânduri e ≈ 0,4 s pe antrenare (optimistă: pe seturi mari exponentul urcă spre 2), adică încă rezonabil pentru 145 de antrenări în căutare.
- **Alternativa** (`Nystroem` cu 300 de componente + `LinearSVC`): PR-AUC 0,565 față de 0,606 la SVC. Pierdem ≈ 0,04 (sub o abatere standard, 0,063). Pe setul nostru nu câștigăm timp, pentru că e prea mic. La 10× sau 100× mai multe rânduri, `Nystroem` crește liniar cu n, iar SVC-ul pătratic, deci acolo aș alege alternativa. Și mai simplu: `LinearSVC` singur are PR-AUC 0,604 (capitolul 8), practic egal cu RBF-ul, și e cel mai rapid.
```

---

## TODO 6 — Probabilități, prag și reziduuri (celula 53) ❌

**Criteriul pentru prag:** la o campanie, a rata un client interesat înseamnă o vânzare pierdută, iar un contact inutil costă puțin. Aleg pragul cu **cea mai mare precizie, cu condiția ca recall ≥ 0,75** (vrem să găsim cel puțin 3 din 4 clienți interesați). Pragul îl aleg pe predicții **out-of-fold** din train, nu pe test.

### Cod corect
```python
# TODO (clasificare): probabilitati din SVM + prag ales FARA test
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss, average_precision_score, precision_score, recall_score, f1_score

fig, ax = plt.subplots(figsize=(5.5, 5))
probas = {}
for method in ["sigmoid", "isotonic"]:
    cal = CalibratedClassifierCV(best_svc, method=method, cv=3)           # calibrare cu CV interna, in fiecare fold
    probas[method] = cross_val_predict(cal, Xc_train, yc_train, cv=cv_c, method="predict_proba")[:, 1]
    print(f"{method:8s}: Brier = {brier_score_loss(yc_train, probas[method]):.4f} | "
          f"PR-AUC = {average_precision_score(yc_train, probas[method]):.3f}")
    CalibrationDisplay.from_predictions(yc_train, probas[method], n_bins=8, name=method, ax=ax)
ax.set_title("Reliability diagram (out-of-fold, train)")
plt.show()

# pragul: precizia maxima cu recall >= 0.75
p = probas["sigmoid"]
MIN_RECALL = 0.75
thresholds = np.arange(0.05, 0.96, 0.01)
table_t = pd.DataFrame({"threshold": thresholds,
                        "recall": [recall_score(yc_train, p >= t) for t in thresholds],
                        "precision": [precision_score(yc_train, p >= t, zero_division=0) for t in thresholds],
                        "F1": [f1_score(yc_train, p >= t) for t in thresholds]})
best_t = table_t[table_t["recall"] >= MIN_RECALL].sort_values("precision").iloc[-1]
print(f"chosen threshold {best_t['threshold']:.2f}: recall {best_t['recall']:.3f}, "
      f"precision {best_t['precision']:.3f}, F1 {best_t['F1']:.3f}  (F1 at 0.50: {f1_score(yc_train, p >= 0.5):.3f})")
```

```python
# TODO (regresie): erorile SVR-ului, pe predictii out-of-fold
from sklearn.metrics import mean_absolute_error

best_svr = grid_svr.best_estimator_
pred_oof = cross_val_predict(best_svr, Xr_train, yr_train, cv=cv_r)
resid = yr_train - pred_oof
print(f"out-of-fold MAE = {mean_absolute_error(yr_train, pred_oof):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].scatter(yr_train, pred_oof, s=10, alpha=0.5)
lims = [0, yr_train.max()]
axes[0].plot(lims, lims, "k--", linewidth=1)
axes[0].set(xlabel=f"real {TARGET_REG}", ylabel="predicted (out-of-fold)", title="Predicted vs. real")
axes[1].scatter(pred_oof, resid, s=10, alpha=0.5)
axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
axes[1].set(xlabel="predicted", ylabel="residual = real - predicted", title="Residuals")
plt.tight_layout()
plt.show()

# eroarea pe intervale ale tintei: unde greseste cel mai mult?
bins = pd.qcut(yr_train, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
display(pd.DataFrame({"MAE": resid.abs(), "mean residual": resid}).groupby(bins, observed=True).mean().round(1))

# cei mai prost prezisi clienti
worst = resid.abs().sort_values(ascending=False).index[:8]
display(Xr_train.loc[worst, ["Income", "Kidhome", "Teenhome", "Year_Birth", "NumWebVisitsMonth"]]
        .assign(real=yr_train[worst], predicted=pred_oof[Xr_train.index.get_indexer(worst)].round(0)))
```

### Observații
```markdown
**Observații:**
- **Calibrare:** ambele metode dau scoruri Brier aproape egale (sigmoid 0,0884, isotonic 0,0890) și stau aproape de diagonală pe diagrama de fiabilitate. Aleg **sigmoid** (Platt): e mai stabilă pe un set mic (la noi ≈ 250 de pozitive în train) și păstrează ordinea scorurilor, deci PR-AUC rămâne același (0,593 față de 0,585 la isotonic).
- **Pragul:** **0,19**. Cu el modelul găsește 75% dintre clienții care acceptă oferta (recall 0,752), iar 44% dintre clienții contactați chiar acceptă (precizie 0,440). Pragul implicit 0,5 ar da F1 0,508 și un recall mult mai mic. Pentru marketing: contactăm toți clienții cu probabilitate de răspuns ≥ 19%.
- **Regresie, reziduuri:** se vede o **pâlnie**: erorile cresc cu valoarea țintei (MAE 90 în primul sfert al țintei, 323 în ultimul). Modelul **subestimează** clienții care cheltuie mult (reziduul mediu +189 în Q4) și supraestimează ușor clienții mici (−61 în Q1, Q2). Cei mai prost prezisi sunt clienți cu venit mare (80k–157k), fără copii, la care modelul prezice 1400–2300, dar care au cheltuit foarte puțin (6, 8, 523). Venitul mare nu garantează cheltuieli mari, iar fără `Num*Purchases` modelul nu are de unde ști cât de activ e clientul. Pâlnia sugerează că o transformare **logaritmică a țintei** ar ajuta (o testăm în L05).
```

---

## TODO 7 — Ce a învățat SVM-ul (celula 58) ❌

### Cod corect
```python
# TODO: ponderile unui SVM liniar + numarul de vectori suport al celui mai bun RBF
from sklearn.svm import LinearSVR

lin_c = make_pipe(LinearSVC(C=0.1, class_weight="balanced", max_iter=10000), num_cols_c).fit(Xc_train, yc_train)
w_c = pd.Series(lin_c["model"].coef_[0], index=lin_c["prep"].get_feature_names_out())
lin_r = TransformedTargetRegressor(make_pipe(LinearSVR(C=1, max_iter=10000), num_cols_r),
                                   transformer=StandardScaler()).fit(Xr_train, yr_train)
w_r = pd.Series(lin_r.regressor_["model"].coef_, index=lin_r.regressor_["prep"].get_feature_names_out())

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, w, title in [(axes[0], w_c, "LinearSVC: weight > 0 -> more likely to respond"),
                     (axes[1], w_r, "LinearSVR: weight > 0 -> spends more")]:
    top = w.reindex(w.abs().sort_values().index).tail(10)          # cele mai mari 10 ponderi (in valoare absoluta)
    ax.barh(top.index, top.values, color=np.where(top.values > 0, "tab:red", "tab:blue"))
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set(title=title, xlabel="weight (scaled features)")
plt.tight_layout()
plt.show()

for name, model, n_rows in [("SVC", grid_svc.best_estimator_["model"], len(Xc_train)),
                            ("SVR", grid_svr.best_estimator_.regressor_["model"], len(Xr_train))]:
    n_sv = len(model.support_)
    print(f"best {name} ({model.kernel}): {n_sv} support vectors = {n_sv / n_rows:.0%} of the training rows")
```

### Observații
```markdown
**Observații:**
- **Clasificare:** cele mai mari ponderi (în valoare absolută): `Marital_Status` Married (−0,37) și Together (−0,35), `Recency` (−0,33), `Customer_Days` (+0,30), `Teenhome` (−0,24), `NumStorePurchases` (−0,24), apoi `AcceptedCmp5` (+0,19), `Education` (+0,18), `AcceptedCmp3` (+0,18). Are sens și se potrivește cu L02: clienții care trăiesc în cuplu răspund mai rar (10–11% față de 22% la Single), cine n-a mai cumpărat de mult nu răspunde, clienții vechi și cei care au acceptat campanii anterioare răspund mai des. Nicio pondere nu e „prea mare ca să fie adevărată” (cea mai mare e doar 0,37), deci nu văd leakage.
- **Regresie:** `Income` domină (+0,69, de 3 ori mai mult decât următoarea), apoi `Marital_Status` infrequent (−0,22, dar sunt doar 4 clienți: o pondere instabilă, nu o regulă), `Teenhome` (−0,16), `Kidhome` (−0,15), `Customer_Days` (+0,15), `AcceptedCmp5` (+0,14). Clienții cu venit mare, fără copii și mai vechi cheltuie mai mult. Are sens; după ce am scos `Num*Purchases`, nu mai există o coloană suspectă.
- **Vectori suport:** SVC-ul RBF are 728 de vectori suport (44% din train), iar SVR-ul 1335 (81%). La SVR, cu ε = 0,05 (tub îngust), aproape toate punctele ies din tub; la SVC, aproape jumătate dintre clienți sunt în zona de amestec dintre clase: problema e „grea”.
```

---

## TODO 8 — Tabelul de comparație și testul t corectat (celula 64) ❌

„Cel mai bun model din L03”: regresia logistică cu `class_weight="balanced"`, C = 1. La regresie, L03 nu a avut o țintă de regresie, așa că iau cel mai bun model din L02 (arborele cu `max_depth=5`) plus random forest ca referință neliniară.

### Cod corect
```python
# TODO: tabelul de comparatie + testul t corectat + salvare
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

models_c = {
    "baseline":                    make_pipe(DummyClassifier(strategy="prior"), num_cols_c),
    "logistic regression (L03)":   make_pipe(LogisticRegression(C=1, max_iter=1000, class_weight="balanced"), num_cols_c),
    "random forest (L03 tuned)":   make_pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced",
                                                                    random_state=SEED), num_cols_c, scaler="passthrough"),
    "LinearSVC":                   make_pipe(LinearSVC(C=0.1, class_weight="balanced", max_iter=10000), num_cols_c),
    "SVM (best, ch. 4)":           grid_svc.best_estimator_,
}
models_r = {
    "baseline (mean)":             make_pipe(DummyRegressor(), num_cols_r),
    "linear regression":           make_pipe(LinearRegression(), num_cols_r),
    "tree (depth 5, L02)":         make_pipe(DecisionTreeRegressor(max_depth=5, random_state=SEED), num_cols_r, scaler="passthrough"),
    "random forest":               make_pipe(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED),
                                             num_cols_r, scaler="passthrough"),
    "LinearSVR":                   TransformedTargetRegressor(make_pipe(LinearSVR(C=1, max_iter=10000), num_cols_r),
                                                              transformer=StandardScaler()),
    "SVR (best, ch. 4)":           grid_svr.best_estimator_,
}

def compare(models, X, y, task, cv):
    rows, folds = [], {}
    for name, model in models.items():
        row, folds[name] = evaluate(model, X, y, task, cv)
        rows.append({"task": task, "model": name, **row})
    main = next(iter(METRICS[task]))
    table = pd.DataFrame(rows).set_index("model").sort_values(main, ascending=(task == "regression"))
    return table, folds, main

table_c, folds_c, main_c = compare(models_c, Xc_train, yc_train, "classification", cv_c)
table_r, folds_r, main_r = compare(models_r, Xr_train, yr_train, "regression", cv_r)
display(table_c.round(3))
display(table_r.round(3))

# testul t corectat: SVM / SVR vs. cel mai bun model care NU e SVM (tabelele sunt sortate)
for table, folds, X, task, svm_name in [(table_c, folds_c, Xc_train, "classification", "SVM (best, ch. 4)"),
                                        (table_r, folds_r, Xr_train, "regression", "SVR (best, ch. 4)")]:
    other = [m for m in table.index if "SV" not in m and "baseline" not in m][0]
    k = len(folds[svm_name])
    t, p_val = corrected_ttest(folds[svm_name], folds[other], len(X) * (k - 1) / k, len(X) / k)
    diff = folds[svm_name].mean() - folds[other].mean()
    print(f"{task:14s}: {svm_name} vs {other}: diff = {diff:+.3f}, t = {t:.2f}, p = {p_val:.3f} -> "
          f"{'significant' if p_val < 0.05 else 'NOT significant'}")

pd.concat([table_c, table_r]).to_csv(f"{RESULTS_DIR}/L04_{MY_NAME}_cv_results.csv")
joblib.dump(grid_svc.best_estimator_, f"{MODELS_DIR}/L04_{MY_NAME}_svc.joblib")
joblib.dump(grid_svr.best_estimator_, f"{MODELS_DIR}/L04_{MY_NAME}_svr.joblib")
```

### Observații
```markdown
**Observații:**
- **Clasificare:** SVM-ul RBF (PR-AUC 0,606), `LinearSVC` (0,604) și regresia logistică din L03 (0,601) sunt practic la egalitate; diferențele de 0,005 sunt de 13 ori mai mici decât abaterea standard (0,065). Random forest-ul din L03 e puțin sub (0,568), cu overfitting mare (0,908 pe train). Varianta liniară este la fel de bună ca RBF-ul.
- **Testul t corectat (clasificare):** SVM vs. regresia logistică: diff = +0,005, p = 0,845 → **nesemnificativ**. Păstrez **regresia logistică**: același scor, de 2 ori mai rapidă, dă direct probabilități (fără calibrare) și ponderi ușor de explicat.
- **Regresie:** SVR (MAE 199) bate clar modelele liniare (233, 229) și arborele (217), dar random forest-ul e puțin mai bun (192). SVR vs. random forest: diff = +7,0, p = 0,130 → **nesemnificativ**. Aș păstra random forest-ul: e la fel de bun sau mai bun și nu cere scalarea țintei.
- **Unde a ajutat mai mult SVM-ul:** la **regresie**: RBF-ul câștigă ≈ 34 la MAE față de varianta liniară (de 3 ori abaterea), pe când la clasificare câștigul e zero. Relația dintre atribute și cheltuieli e neliniară (venitul are efect mai mare la venituri mari, copiii scad cheltuielile brusc), iar kernelul o prinde. Răspunsul la campanie depinde în mare parte liniar de atribute (campanii anterioare, recență), deci acolo un model liniar ajunge.
```

---

## TODO 9 — Găsiți greșelile (celula 70) ❌

### Ce e greșit în rezolvarea ta
1. **Alt set de date:** citești `WA_Fn-UseC_-HR-Employee-Attrition.csv` (IBM HR Attrition), care nu există în Drive-ul tău. Celula dă `FileNotFoundError`. TODO-ul cere explicit **pe setul vostru**.
2. **Alt model:** blocul din enunț folosește **`SVC`**; tu ai `LogisticRegression`.
3. **„Corectezi” greșeli care nu sunt în bloc:** *„LogisticRegression în loc de LinearRegression”* (blocul nu are `LinearRegression`), *„StratifiedKFold în loc de KFold simplu”* (blocul are `cv=5`), *„scoring=roc_auc în loc de accuracy”* (blocul are `scoring="f1"`). Pare o rezolvare a altui exercițiu, adusă aici.
4. **Lipsesc corecturile reale:** `sklearn.metric`, `predict_proba` pe SVC, `f1_score` pe probabilități, MAE la clasificare, grila liniară C/γ.
5. Celula de observații (71) e goală.

### Cod corect
```python
# TODO: blocul corectat, pe setul nostru (clasificare: Response)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score   # corectat: sklearn.metrics (nu .metric); fara mean_absolute_error

X_err = df.drop(columns=[TARGET_CLF, TARGET_REG])
y_err = (df[TARGET_CLF] == POS_CLASS).astype(int)

# corectat: stratify=y - clasa 1 are doar 15%, fara stratificare proportia difera intre train si test
X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(X_err, y_err, test_size=0.2, stratify=y_err, random_state=SEED)
# corectat: testul ramane inchis in L04 -> "evaluarea finala" se face pe o parte scoasa din train
X_fit_e, X_hold_e, y_fit_e, y_hold_e = train_test_split(X_tr_e, y_tr_e, test_size=0.2, stratify=y_tr_e, random_state=SEED)

# corectat: scalarea (si imputarea + encodarea categoriilor, fara de care SVC crapa pe NaN / text) e IN pipeline,
#           deci se invata doar din partea de antrenare a fiecarui fold, nu din tot X (leakage)
# corectat: class_weight="balanced" - fara el SVC-ul prezice mereu 0 pe o clasa de 15% (F1 = 0)
pipe_e = make_pipe(SVC(class_weight="balanced"), X_err.select_dtypes(include="number").columns.tolist())

# corectat: C si gamma pe scara LOGARITMICA (1, 2, 3 sunt aproape aceeasi valoare; gamma 1-3 = "insule" -> overfitting)
param_grid_e = {"model__C": np.logspace(-1, 2, 4), "model__gamma": np.logspace(-3, 0, 4)}
# corectat: fold-uri stratificate si amestecate, aceleasi pentru toate combinatiile
cv_e = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_e = GridSearchCV(pipe_e, param_grid_e, cv=cv_e, scoring="f1", n_jobs=-1)
grid_e.fit(X_fit_e, y_fit_e)          # corectat: cautarea se face pe datele de ANTRENARE, nu pe test

# corectat: SVC nu are predict_proba (probability=False) -> clasele cu predict, scorul cu decision_function
pred_e = grid_e.predict(X_hold_e)
score_e = grid_e.decision_function(X_hold_e)
print("best:", grid_e.best_params_, f"| CV F1 = {grid_e.best_score_:.3f}")
print(f"holdout F1 = {f1_score(y_hold_e, pred_e):.3f}")          # corectat: F1 pe CLASE prezise, nu pe probabilitati
print(f"holdout PR-AUC = {average_precision_score(y_hold_e, score_e):.3f}")
# corectat: am scos MAE - este o metrica de REGRESIE, nu are sens pentru clase 0/1
```

### Observații (celula 71)
```markdown
**Greșelile găsite:**
1. `from sklearn.metric import ...` → `sklearn.metrics`. **Eroare** (`ModuleNotFoundError`).
2. `train_test_split` fără `stratify=y` la o clasă rară (15%): proporția clasei 1 poate diferi între train și test. **Fără eroare.**
3. `StandardScaler().fit(X)` pe **tot** setul, inclusiv testul: media și abaterea „văd” testul (leakage). Corect: scalarea în pipeline. **Fără eroare.**
4. Lipsesc imputarea și encodarea categoriilor: pe setul nostru, `SVC` primește NaN (`Income`) și text (`Education`). **Eroare** (`ValueError`).
5. Grila `C = [1, 2, 3]`, `gamma = [1, 2, 3]` e liniară și îngustă: C și γ au efect multiplicativ, deci se caută pe scară logaritmică; γ ≥ 1 dă „insule” (overfitting). **Fără eroare.**
6. `SVC()` fără `class_weight="balanced"` pe o clasă de 15%: poate prezice mereu 0 (F1 = 0). **Fără eroare.**
7. `cv=5`: fold-uri neamestecate; mai bine același `StratifiedKFold(shuffle=True)` ca în rest. **Fără eroare.**
8. `grid.fit(X_test_s, y_test)`: căutarea **pe test**, deci testul nu mai e independent; scorul raportat e prea optimist. **Fără eroare.**
9. `predict_proba` pe un `SVC` cu `probability=False`. **Eroare** (`AttributeError`). Corect: `decision_function` sau `CalibratedClassifierCV`.
10. `f1_score(y_test, proba)`: F1 se calculează pe clase, nu pe probabilități. **Eroare** (`ValueError`: mix de țintă binară și continuă).
11. `mean_absolute_error` pe o clasificare: metrică de regresie, fără sens pentru 0/1. **Fără eroare.**

**Cele mai periculoase sunt cele fără eroare** (2, 3, 5, 6, 7, 8, 11): codul rulează și produce un număr care pare bun. Leakage-ul (3) și căutarea pe test (8) umflă scorul, iar nimic nu te avertizează. Erorile care crapă le repari oricum, pentru că nu poți continua.
```

---

## TODO 10 — Contribuția (celula 73) ❌

**Ipoteză:** atributele agregate (numărul total de campanii acceptate, numărul de copii, vârsta, cheltuiala pe o cumpărătură) ajută SVM-ul RBF, pentru că rezumă informație împrăștiată în mai multe coloane. Kernelul RBF măsoară distanțe, iar 5 coloane `AcceptedCmp*` cântăresc în distanță cât 5 atribute separate, deși spun un singur lucru („cât de receptiv e clientul la campanii”).

### Cod corect
```python
# CONTRIBUTIA: atribute noi pentru SVM (clasificare), pe ACELEASI fold-uri, cu testul t corectat
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone

ACC_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

def add_features(X):
    X = X.copy()
    X["AcceptedTotal"] = X[ACC_COLS].sum(axis=1)                         # cate campanii anterioare a acceptat
    X["Children"] = X["Kidhome"] + X["Teenhome"]
    X["Age"] = 2014 - X["Year_Birth"]
    X["SpendPerPurchase"] = X[MNT_COLS].sum(axis=1) / X[PURCHASE_COLS].sum(axis=1).clip(lower=1)
    return X.drop(columns=ACC_COLS + ["Kidhome", "Teenhome", "Year_Birth"])

new_num = [c for c in add_features(Xc_train.head()).columns if c not in cat_cols_c]
svc_new = Pipeline([("features", FunctionTransformer(add_features)),       # fara parametri invatati -> fara leakage
                    ("svm", make_pipe(clone(grid_svc.best_estimator_["model"]), new_num))])

row_old, f_old = evaluate(grid_svc.best_estimator_, Xc_train, yc_train, "classification", cv_c)
row_new, f_new = evaluate(svc_new, Xc_train, yc_train, "classification", cv_c)
display(pd.DataFrame([row_old, row_new], index=["SVM, original features", "SVM, + aggregated features"]).round(3))
k = len(f_old)
t, p_val = corrected_ttest(f_new, f_old, len(Xc_train) * (k - 1) / k, len(Xc_train) / k)
print(f"diff = {f_new.mean() - f_old.mean():+.3f}, t = {t:.2f}, p = {p_val:.3f}")
```

### Observații
```markdown
**Contribuția:**
- **Ipoteza:** atributele agregate (`AcceptedTotal`, `Children`, `Age`, `SpendPerPurchase`) ajută SVM-ul RBF, pentru că rezumă informație împrăștiată în mai multe coloane, care în distanța kernelului cântăreau de mai multe ori.
- **Experimentul:** același SVC (C = 100, γ = 0,001, `class_weight="balanced"`), aceleași 5 fold-uri, cu și fără atributele noi; atributele se calculează într-un `FunctionTransformer`, fără nimic învățat din date, deci fără leakage.
- **Rezultatul:** PR-AUC 0,606 → 0,625 (+0,020), dar F1 0,551 → 0,540 și recall 0,776 → 0,756. Testul t corectat: t = 2,22, p = 0,090.
- **Concluzia:** atributele noi îmbunătățesc ordonarea clienților (PR-AUC crește la fiecare fold, de aceea t e mare), dar câștigul nu e semnificativ la p < 0,05 cu doar 5 fold-uri, iar la pragul implicit F1 scade ușor. Ipoteza e susținută doar parțial. Merită testat cu fold-uri repetate (`RepeatedStratifiedKFold`) și la modelele din L05, unde atributele agregate pot ajuta mai mult arborii.
```

---

## Temă pentru laboratorul 5 (celula 76) ❌

```markdown
| Ce fac | Clasa / funcția | Modulul | Pasul |
|---|---|---|---|
| împart datele train / test | `train_test_split` | `sklearn.model_selection` | split |
| validare încrucișată | `cross_validate`, `cross_val_score`, `StratifiedKFold`, `KFold` | `sklearn.model_selection` | split + fit + metrică (evaluare) |
| căutarea hiperparametrilor | `GridSearchCV`, `RandomizedSearchCV` | `sklearn.model_selection` | instanțiere + fit |
| completez valorile lipsă | `SimpleImputer` | `sklearn.impute` | preprocesare |
| scalez / codific | `StandardScaler`, `OneHotEncoder` | `sklearn.preprocessing` | preprocesare |
| pași diferiți pe coloane diferite | `ColumnTransformer` | `sklearn.compose` | preprocesare |
| leg pașii într-un lanț | `Pipeline`, `make_pipeline` | `sklearn.pipeline` | instanțiere |
| SVM pentru clasificare / regresie | `SVC`, `LinearSVC`, `SVR` | `sklearn.svm` | instanțiere + fit + predict |
| scalez ținta | `TransformedTargetRegressor` | `sklearn.compose` | preprocesare (a țintei) |
| arbore, regresie logistică, kNN | `DecisionTreeClassifier`, `LogisticRegression`, `KNeighborsClassifier` | `sklearn.tree`, `sklearn.linear_model`, `sklearn.neighbors` | instanțiere + fit + predict |
| metrici de clasificare | `f1_score`, `roc_auc_score`, `average_precision_score`, `confusion_matrix` | `sklearn.metrics` | metrică |
| metrici de regresie | `mean_absolute_error`, `root_mean_squared_error`, `r2_score` | `sklearn.metrics` | metrică |
| salvez / încarc un model | `dump`, `load` | `joblib` | după fit (salvare) / înainte de predict (încărcare) |

1. **`predict_proba`** dă probabilități. `SVC` (cu `probability=False`, implicit), `LinearSVC` și `RidgeClassifier` nu o au. În loc: `decision_function` (un scor, suficient pentru ROC-AUC / PR-AUC) sau `CalibratedClassifierCV`, care transformă scorul în probabilitate.
2. **`StandardScaler` se face `fit` doar pe datele de antrenare** (în CV: pe partea de antrenare a fiecărui fold). Media și abaterea standard sunt „învățate”; dacă le calculăm și pe test, testul influențează modelul (leakage), iar scorul iese prea optimist.
3. **`predict` = `predict_proba(X)[:, 1] >= 0,5`** la un clasificator binar care are probabilități: `predict` întoarce clasa (0/1), `predict_proba` întoarce probabilitatea; `predict` aplică pragul fix 0,5. Cu `predict_proba` putem alege alt prag.
```

---

## Celulele AI ⚠️

Toate sunt goale. Dacă ai folosit un asistent (inclusiv acest raport), completează-le cu promptul, ce ai primit și **cum ai verificat** (de exemplu: „am rulat celula și am comparat scorul regresiei logistice cu L03: aceeași valoare”).

## Final (celula 78)

După rulare trebuie să existe `results/L04_<MY_NAME>_cv_results.csv`, `models/L04_<MY_NAME>_svc.joblib` și `models/L04_<MY_NAME>_svr.joblib`.
