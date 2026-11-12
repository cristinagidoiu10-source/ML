# Teorie — Laboratorul 4: SVM și SVR

> Adaptat după `lab4/TEORIE_LAB4.md` din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB) (Telco), refăcut pe **Customer Personality Analysis**. Cifrele vin din codul corectat din `L04_raport.md` (CV pe train, `clean.csv`).

Din Lab 4 lucrăm cu **ambele probleme** în paralel:
- **clasificare:** `Response` (acceptă oferta?) — X fără `MntTotal`;
- **regresie:** `MntTotal` (cât cheltuie în 2 ani) — X fără `Mnt*`, `Num*Purchases`, `Response` (leakage, Lab 2).

Modelul nou: **mașina cu vectori suport** (SVM) pentru clasificare și varianta ei pentru regresie (SVR). La final (secțiunea 15) e **codul explicat**.

**Cuprins**
1. Metricile alese în Lab 4
2. Ideea SVM
3. Vectorii suport
4. Marginea „soft” și C
5. Kernelul și γ
6. SVR: tubul ε
7. Scalarea la SVM / SVR
8. Scoruri, probabilități, calibrare
9. Predicții out-of-fold și pragul
10. Reziduurile
11. Viteza
12. Ce a învățat modelul
13. Compararea modelelor: testul t corectat
14. Model degenerat
15. **Codul din Lab 4, explicat**
16. Rezumat + întrebări

---

## 1. Metricile alese în Lab 4

### Metrica principală
Se alege **înainte** de a vedea rezultatele (altfel alegi metrica la care iese bine modelul preferat).

### Clasificare → PR-AUC (`average_precision`)
**Curba precizie–recall:** pentru fiecare prag posibil, un punct (recall, precizie). **PR-AUC** = aria de sub ea.
**De ce la noi:** clasa 1 are doar 15%. PR-AUC măsoară cât de bine **ordonează** modelul clienții după șansa de răspuns — exact ce face marketingul (contactăm întâi scorurile mari) — și nu depinde de prag. Baseline-ul are PR-AUC ≈ **0,15** (proporția clasei 1), nu 0,85 ca acuratețea.
Metrici secundare: **F1** și **recall** (ca să prindem un model degenerat), acuratețe echilibrată.

### ROC-AUC (recapitulare)
Aria sub curba (rata FP, recall). Baseline = 0,5. La clase foarte dezechilibrate e optimistă (TN sunt mulți și ușor de obținut); PR-AUC e mai severă.

### Metricile de regresie
| Metrică | Ce înseamnă | La noi |
|---|---|---|
| **MAE** | eroarea medie absolută, în unități monetare | **metrica principală**: „greșim în medie cu X pe client” |
| RMSE | rădăcina erorii pătratice medii; penalizează mult erorile mari | dominată de cei câțiva clienți care cheltuie > 2000 |
| R² | ce fracțiune din variație explică modelul (1 = perfect, 0 = cât media) | |

**Referințe (CV pe train):** regresie logistică PR-AUC **0,601 ± 0,065**, F1 0,562; regresie liniară MAE **233**, R² 0,715.

---

## 2. Ideea SVM

**Ce înseamnă:** dintre toate dreptele (hiperplanele) care separă cele două clase, SVM o alege pe cea cu **marginea cea mai largă** — cea mai depărtată de cele mai apropiate puncte din fiecare clasă. O margine largă = un model mai robust pe date noi.

**La noi (desen pe 2 atribute: `Recency` și `MntWines`):** clienții care acceptă au cumpărat recent și cheltuie mult pe vin; SVM caută dreapta care îi separă cel mai bine de ceilalți.

---

## 3. Vectorii suport

**Ce înseamnă:** punctele care stau **pe margine sau în interiorul ei** (sau de partea greșită). Doar ele determină frontiera; celelalte puncte pot fi mutate fără să schimbe modelul.
**La noi:** SVM-ul final (RBF) are **728 de vectori suport = 44% din train** → aproape jumătate dintre clienți sunt în zona de amestec dintre clase: problema e „grea”. SVR-ul are 1335 (81%).

---

## 4. Marginea „soft” și hiperparametrul C

**Ce înseamnă:** datele reale nu sunt perfect separabile. **Marginea soft** permite unor puncte să fie în margine sau de partea greșită, cu o penalizare. **C** = cât de scumpă e o încălcare:
- **C mic** → margine largă, multe încălcări permise, model simplu (risc de underfitting);
- **C mare** → margine îngustă, încearcă să clasifice corect fiecare punct (risc de overfitting).

**La noi (SVC liniar, `Recency` + `MntWines`, 400 de clienți):**
| C | Lățimea marginii (scalată) | Vectori suport |
|---|---|---|
| 0,01 | 3,04 | 337 / 400 |
| 1 | 2,07 | 287 |
| 100 | 2,07 | ~287 (nu se mai schimbă) |

Clasele **nu sunt separabile liniar** în aceste două atribute: chiar cu C = 100, acuratețea rămâne ≈ 0,68.

---

## 5. Kernelul și γ (gamma)

### Problema
O dreaptă nu poate separa clase care se amestecă în forme curbe.

### Trucul kernel
**Ce înseamnă:** SVM calculează doar **asemănări** (produse scalare) între puncte. Kernelul înlocuiește produsul scalar cu o altă funcție de asemănare → echivalent cu a lucra într-un spațiu cu mai multe dimensiuni, unde clasele devin separabile, **fără** să calculăm explicit acele dimensiuni.
- `linear`: dreaptă / hiperplan;
- `rbf` (gaussian): asemănare = exp(−γ · distanța²) → frontiere curbe;
- `poly`: polinomial (lent, l-am lăsat afară).

### γ (gamma) la RBF
**Ce înseamnă:** cât de „locală” e influența unui punct.
- **γ mic** → fiecare punct influențează o zonă largă → frontieră netedă, aproape liniară;
- **γ mare** → influență foarte locală → „insule” în jurul fiecărui punct → overfitting.

### C și γ se caută împreună
**La noi:** cel mai bun SVC: **RBF, C = 100, γ = 0,001** → PR-AUC **0,606 ± 0,063**. Pe harta C × γ zona bună e o **diagonală** (C mai mare cere γ mai mic). Pentru γ ≥ 0,1: train 0,79–0,98, validare doar 0,44–0,50 → overfitting. Un γ atât de mic (0,001) face frontiera aproape liniară → kernelul liniar (C = 0,01) dă practic același scor (0,599). **Problema de clasificare e în mare parte liniară.**

---

## 6. SVR: regresia cu „tub” ε

**Ce înseamnă:** SVR caută o funcție care să treacă prin date astfel încât cele mai multe puncte să fie **în interiorul unui tub de lățime ±ε**. Erorile mai mici decât ε **nu se penalizează deloc**; doar punctele din afara tubului devin vectori suport și contează.
- **ε mic** → tub îngust → aproape toate punctele contează → model detaliat;
- **ε mare** → tub lat → puține puncte contează → model simplu, „plat”.

**La noi (`Income` → `MntTotal`, ținta scalată, 400 de clienți):**
| ε (în std ale țintei) | Tub în bani | Vectori suport |
|---|---|---|
| 0,1 | ±62 | 346 / 400 |
| 0,5 | ±308 | 139 |
| 1,0 | ±615 | 38 |

**Cel mai bun SVR:** RBF, C = 100, γ = 0,01, ε = 0,05 → **MAE 199,1 ± 10,9** (liniar: ~229). Aici RBF-ul chiar ajută: relația cheltuieli–atribute e neliniară.

---

## 7. Scalarea la SVM / SVR

### Scalarea atributelor e obligatorie
SVM lucrează cu **distanțe**. Nescalat, `Income` (zeci de mii) domină complet `Kidhome` (0–2).
**La noi:** SVC RBF fără scalare: PR-AUC **0,307** (abia peste baseline 0,152); cu `StandardScaler`: **0,567**.

### Scalarea țintei la SVR (`TransformedTargetRegressor`)
`ε = 0,1` și `C = 1` (implicite) sunt gândite pentru o țintă de ordinul 1. `MntTotal` are media ~614 și std ~600: un tub de ±0,1 e minuscul, iar penalizarea e prea mică → SVR rămâne aproape constant.
**La noi:** SVR cu X scalat, dar y nescalat: **R² = 0,01** (MAE 456 — cât baseline-ul). Cu X **și** y scalate: **MAE 194, R² 0,786** (mai bine decât regresia liniară).
`TransformedTargetRegressor(regressor=..., transformer=StandardScaler())` scalează y la `fit` și **de-scalează** automat predicțiile.

---

## 8. Scoruri, probabilități și calibrare

### `decision_function` vs `predict_proba`
- `decision_function`: distanța (cu semn) față de frontieră. Nu e o probabilitate, dar **ordonează** corect → ajunge pentru ROC-AUC / PR-AUC.
- `predict_proba`: probabilitate. `SVC` o are doar cu `probability=True` (sau prin calibrare).

### Calibrare
**Ce înseamnă:** transformăm scorul într-o probabilitate „cinstită”: dintre clienții cu probabilitate 0,3, chiar ~30% să accepte.
- `CalibratedClassifierCV(method="sigmoid")` (Platt) — potrivește o sigmoidă; stabilă pe date puține;
- `method="isotonic"` — funcție în trepte; flexibilă, cere mai multe date.

### Diagrama de fiabilitate
Probabilitatea prezisă (x) vs proporția reală de 1 (y). Calibrare perfectă = diagonala.

### Scorul Brier
Media (probabilitate − realitate)². Mai mic = mai bine.
**La noi:** sigmoid **0,0884**, isotonic 0,0890 → aproape egale; aleg **sigmoid** (≈ 250 de pozitive în train — puține pentru isotonic).

---

## 9. Predicții out-of-fold și pragul

### Predicții out-of-fold (OOF)
`cross_val_predict` → pentru fiecare client din train, predicția modelului care **nu l-a văzut**. Le folosim ca să alegem pragul / să analizăm erorile **fără test**.

### Alegerea pragului
**Criteriul (din povestea campaniei):** cea mai mare precizie, cu condiția **recall ≥ 0,75** (găsim cel puțin 3 din 4 clienți interesați).
**La noi:** prag **0,19** → recall 0,752, precizie 0,440. Pragul 0,5 ar da F1 0,508 și recall mult mai mic.
**Pe înțelesul marketingului:** contactăm toți clienții cu probabilitate de răspuns ≥ 19%; ~44% dintre ei acceptă.

---

## 10. Reziduurile

**Reziduu** = real − prezis. Le desenăm față de predicție: ideal, un nor fără formă în jurul lui 0.
**La noi (SVR, OOF):** o **pâlnie** — erorile cresc cu ținta.
| Sfertul țintei | MAE | Reziduu mediu |
|---|---|---|
| Q1 (cheltuie puțin) | 90 | −61 (supraestimat) |
| Q4 (cheltuie mult) | **323** | **+189** (subestimat) |

Cei mai prost prezisi: venit mare (80k–157k), fără copii, model prezice 1400–2300, dar au cheltuit 6–523. Venitul nu garantează cheltuiala. Pâlnia sugerează **log1p pe țintă** (testat în Lab 5).

---

## 11. Viteza: SVM pe date mari

### Complexitatea
SVC cu kernel crește între **n² și n³** cu numărul de rânduri (calculează asemănări între perechi).
**La noi:** o antrenare pe 1646 de rânduri ≈ **0,06 s**; la 10× → estimat ~0,4 s (optimist). Încă OK.

### Alternative rapide
- `LinearSVC`: liniar, foarte rapid. **La noi:** PR-AUC 0,604 — practic egal cu RBF;
- `Nystroem` + `LinearSVC`: aproximează kernelul RBF cu 300 de componente, crește liniar cu n. La noi: 0,565 (pierdere ~0,04, sub o std).

---

## 12. Ce a învățat modelul

### Ponderile unui SVM liniar
Pe atribute scalate, ponderea arată **direcția** și **mărimea** efectului.
**Clasificare (LinearSVC):** `Marital_Status` Married (−0,37), Together (−0,35), `Recency` (−0,33), `Customer_Days` (+0,30), `Teenhome` (−0,24), `AcceptedCmp5` (+0,19), `Education` (+0,18). Cuplurile, cei care n-au mai cumpărat de mult și cei cu adolescenți răspund mai rar; clienții vechi și cei care au acceptat campanii răspund mai des. Nicio pondere „prea mare” → fără leakage.
**Regresie (LinearSVR):** `Income` domină (**+0,69**), apoi `Teenhome` (−0,16), `Kidhome` (−0,15), `Customer_Days` (+0,15), `AcceptedCmp5` (+0,14).

### Numărul de vectori suport
Mulți vectori suport = mult amestec între clase / tub îngust (secțiunea 3).

---

## 13. Compararea modelelor: testul t corectat

### Problema
Două modele au PR-AUC 0,606 și 0,601. Diferența e reală sau noroc?

### Testul t și p-valoarea
Comparăm scorurile **pe aceleași fold-uri**. **p** = probabilitatea să vedem o diferență cel puțin atât de mare dacă modelele ar fi de fapt egale. **p < 0,05** → diferență semnificativă.

### De ce „corectat” (Nadeau–Bengio)
Fold-urile de antrenare se suprapun (fiecare pereche împarte 3/4 din date) → scorurile nu sunt independente → testul t obișnuit e prea „încrezător”. Corecția mărește varianța cu factorul (1/k + n_test/n_train).

### Briciul lui Occam
La scor egal, alegem modelul **mai simplu**.

**La noi:**
| Problemă | Comparație | Diferență | p | Decizie |
|---|---|---|---|---|
| clasificare | SVM RBF vs regresie logistică | +0,005 PR-AUC | **0,845** | nesemnificativ → păstrăm **regresia logistică** (mai rapidă, probabilități directe, ponderi ușor de explicat) |
| regresie | SVR vs random forest | +7 MAE (SVR mai slab) | 0,130 | nesemnificativ → păstrăm **random forest** (MAE 192, nu cere scalarea țintei) |

---

## 14. Model degenerat

**Ce înseamnă:** un model care prezice (aproape) mereu aceeași clasă. Poate avea PR-AUC/ROC-AUC decent (ordonează bine), dar F1 ≈ 0 la pragul implicit.
**La noi:** SVC RBF scalat, **fără** `class_weight`: PR-AUC 0,567, dar **F1 0,317, recall 0,216**. Cu `class_weight="balanced"`: F1 0,552, recall 0,752. De aceea verificăm mereu F1 / recall pe lângă metrica principală.

---

## 15. Codul din Lab 4, explicat

### 15.1 Cele două probleme

```python
MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
LEAKY_R = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]

Xc = df.drop(columns=[TARGET_CLF, TARGET_REG])
yc = (df[TARGET_CLF] == POS_CLASS).astype(int)
Xr = df.drop(columns=[TARGET_REG, TARGET_CLF] + MNT_COLS + LEAKY_R)
yr = df[TARGET_REG]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=SEED)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)

cv_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_r = KFold(n_splits=5, shuffle=True, random_state=SEED)
```
- Două X-uri **diferite**: la regresie scoatem tot ce conține sau dezvăluie ținta.
- `(df[TARGET_CLF] == POS_CLASS).astype(int)` → ținta sigur 0/1 (True/False → 1/0).
- Clasificarea se stratifică, regresia nu (ținta e continuă). `KFold` simplu la regresie.
- Fold-urile se definesc **o dată** și se refolosesc → toate modelele sunt comparate pe aceleași împărțiri.

### 15.2 Funcția de evaluare

```python
METRICS = {
    "classification": {"PR_AUC": "average_precision", "F1": "f1", "recall": "recall", "bal_acc": "balanced_accuracy"},
    "regression":     {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
}

def evaluate(model, X, y, task, cv):
    scoring = METRICS[task]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    sign = {name: -1 if scorer.startswith("neg_") else 1 for name, scorer in scoring.items()}
    main = next(iter(scoring))
    folds = sign[main] * res[f"test_{main}"]
    row = {main: folds.mean(), f"{main}_std": folds.std(), f"{main}_train": sign[main] * res[f"train_{main}"].mean()}
    row |= {name: sign[name] * res[f"test_{name}"].mean() for name in scoring if name != main}
    row["fit_time_s"] = res["fit_time"].mean()
    return row, folds
```
- `METRICS`: metricile fiecărei probleme într-un singur loc; **prima** din dicționar = metrica principală (`next(iter(scoring))` = prima cheie).
- scikit-learn **maximizează** mereu scorul, așa că erorile vin cu minus (`neg_mean_absolute_error`). `sign` le întoarce la valori pozitive, ușor de citit.
- Întoarce un **rând de tabel** (medie, std, scor pe train, metrici secundare, timp) și **scorurile pe fold-uri** ale metricii principale (pentru testul t).
- `row |= {...}` = adaugă chei noi în dicționar (Python 3.9+).

```python
def make_pipe(model, num_cols, scaler=None):
    num_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler() if scaler is None else scaler)])
    ...
    return Pipeline([("prep", prep), ("model", model)])
```
Aceeași preprocesare ca în L03, dar cu `StandardScaler` implicit (SVM are nevoie de scalare; în L03 Standard și Robust au ieșit la egalitate).

### 15.3 Efectul lui C (desen)

```python
X2, _, y2, _ = train_test_split(Xc_train[["Recency", "MntWines"]], yc_train, train_size=400,
                                stratify=yc_train, random_state=SEED)
for ax, C in zip(axes, [0.01, 1, 100]):
    svm2 = Pipeline([("scale", StandardScaler()),
                     ("model", SVC(kernel="linear", C=C, class_weight="balanced"))]).fit(X2, y2)
    width = 2 / np.linalg.norm(svm2["model"].coef_[0])
    plot_svc_2d(svm2, X2, y2, ax, f"C = {C}: {svm2['model'].n_support_.sum()} support vectors ...")
```
- 2 atribute (ca să putem desena în plan), 400 de puncte (lizibil).
- `zip(axes, [0.01, 1, 100])` = un subgrafic pentru fiecare C.
- Lățimea marginii = **2 / ‖w‖** (teoria SVM): `np.linalg.norm` = lungimea vectorului de ponderi.
- `n_support_` = câți vectori suport are fiecare clasă; `.sum()` = total.

### 15.4 Scalarea obligatorie

```python
for name, scaler, cw in [("SVC rbf, no scaling", "passthrough", "balanced"),
                         ("SVC rbf + StandardScaler", None, None),
                         ("SVC rbf + StandardScaler + class_weight", None, "balanced")]:
    row, _ = evaluate(make_pipe(SVC(kernel="rbf", class_weight=cw), num_cols_c, scaler=scaler), ...)

"SVR rbf + scaling X and y": TransformedTargetRegressor(make_pipe(SVR(kernel="rbf"), num_cols_r),
                                                        transformer=StandardScaler()),
```
Un experiment controlat: schimbăm **un singur lucru** pe rând (scalarea, apoi `class_weight`, apoi scalarea țintei) și vedem efectul în cifre.

### 15.5 GridSearch pentru SVC și SVR

```python
param_grid_svc = [
    {"model__kernel": ["linear"], "model__C": np.logspace(-2, 1, 4)},
    {"model__kernel": ["rbf"], "model__C": np.logspace(-1, 3, 5), "model__gamma": np.logspace(-4, 0, 5)},
]
grid_svc = GridSearchCV(make_pipe(SVC(class_weight="balanced"), num_cols_c), param_grid_svc, cv=cv_c,
                        scoring="average_precision", n_jobs=-1, return_train_score=True)
```
- Grila e o **listă de dicționare** → sub-grile separate: `gamma` se caută doar pentru `rbf` (la `linear` nu are sens). Total 4 + 25 = 29 de combinații.
- `np.logspace(-4, 0, 5)` = 0,0001; 0,001; 0,01; 0,1; 1.

```python
prefix = "regressor__model__"
grid_svr = GridSearchCV(TransformedTargetRegressor(make_pipe(SVR(), num_cols_r), transformer=StandardScaler()),
                        param_grid_svr, cv=cv_r, scoring="neg_mean_absolute_error", ...)
```
Pipeline-ul e **în interiorul** lui `TransformedTargetRegressor` (sub numele `regressor`) → parametrii au prefixul `regressor__model__`.

```python
cg_heatmaps(grid_svc, title="SVC rbf, PR-AUC")
```
Funcția din exemplu: desenează harta C × γ pentru train și validare → vezi zona stabilă și zona de overfitting.

### 15.6 Timpul de antrenare

```python
for n in sizes:
    t0 = time.time()
    clone(best_svc).fit(Xs, ys)
    times.append(time.time() - t0)
slope = np.polyfit(np.log(sizes), np.log(times), 1)[0]
```
- `clone` = o copie **neantrenată** a modelului (aceiași hiperparametri).
- Dacă timp ≈ n^a, atunci log(timp) = a · log(n) + const → panta dreptei din `polyfit` pe logaritmi = exponentul `a`.

```python
alt = make_pipe(Pipeline([("map", Nystroem(kernel="rbf", gamma=gamma, n_components=300, random_state=SEED)),
                          ("svm", LinearSVC(C=0.1, class_weight="balanced", max_iter=10000))]), num_cols_c)
```
`Nystroem` transformă datele în 300 de coloane noi care aproximează kernelul RBF; peste ele, un SVM liniar rapid.

### 15.7 Calibrare și prag

```python
cal = CalibratedClassifierCV(best_svc, method=method, cv=3)
probas[method] = cross_val_predict(cal, Xc_train, yc_train, cv=cv_c, method="predict_proba")[:, 1]
brier_score_loss(yc_train, probas[method])
CalibrationDisplay.from_predictions(yc_train, probas[method], n_bins=8, name=method, ax=ax)
```
- Calibrarea are propria CV internă (`cv=3`) → învață sigmoida pe date nevăzute de SVM.
- Totul în `cross_val_predict` → probabilități OOF pentru fiecare client din train.

```python
table_t = pd.DataFrame({"threshold": thresholds,
                        "recall": [recall_score(yc_train, p >= t) for t in thresholds],
                        "precision": [precision_score(yc_train, p >= t, zero_division=0) for t in thresholds], ...})
best_t = table_t[table_t["recall"] >= MIN_RECALL].sort_values("precision").iloc[-1]
```
Un tabel cu recall/precizie pentru fiecare prag 0,05…0,95; filtrăm pragurile cu recall ≥ 0,75, sortăm după precizie și luăm **ultimul** rând (`iloc[-1]`) = cea mai mare precizie.

### 15.8 Reziduuri

```python
pred_oof = cross_val_predict(best_svr, Xr_train, yr_train, cv=cv_r)
resid = yr_train - pred_oof
bins = pd.qcut(yr_train, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
pd.DataFrame({"MAE": resid.abs(), "mean residual": resid}).groupby(bins, observed=True).mean()
worst = resid.abs().sort_values(ascending=False).index[:8]
```
- `pd.qcut(..., 4)` împarte ținta în **4 sferturi** cu același număr de clienți → MAE pe fiecare sfert arată **unde** greșește modelul.
- `worst` = indicii celor 8 clienți cu eroarea cea mai mare → îi inspectăm ca să înțelegem de ce.

### 15.9 Ponderile modelelor liniare

```python
lin_c = make_pipe(LinearSVC(C=0.1, class_weight="balanced", max_iter=10000), num_cols_c).fit(Xc_train, yc_train)
w_c = pd.Series(lin_c["model"].coef_[0], index=lin_c["prep"].get_feature_names_out())
top = w.reindex(w.abs().sort_values().index).tail(10)
```
- `coef_[0]` = ponderile; `get_feature_names_out()` = numele coloanelor **după** preprocesare (inclusiv cele one-hot) → le punem împreună într-o `Series`.
- Sortăm după valoarea absolută și păstrăm ultimele 10 = cele mai mari efecte (pozitive sau negative).

### 15.10 Tabelul de comparație și testul t

```python
def compare(models, X, y, task, cv):
    rows, folds = [], {}
    for name, model in models.items():
        row, folds[name] = evaluate(model, X, y, task, cv)
        rows.append({"task": task, "model": name, **row})
    main = next(iter(METRICS[task]))
    table = pd.DataFrame(rows).set_index("model").sort_values(main, ascending=(task == "regression"))
    return table, folds, main

t, p_val = corrected_ttest(folds[svm_name], folds[other], len(X) * (k - 1) / k, len(X) / k)
```
- `sort_values(..., ascending=(task == "regression"))`: la regresie MAE mic = bun (crescător), la clasificare PR-AUC mare = bun (descrescător).
- `corrected_ttest` primește scorurile pe fold-uri ale celor două modele + dimensiunile train/validare ale unui fold (pentru corecția Nadeau–Bengio).
- La final, `to_csv` și `joblib.dump` → rezultatele și modelele se refolosesc în L05.

### 15.11 Contribuția: atribute agregate

```python
def add_features(X):
    X = X.copy()
    X["AcceptedTotal"] = X[ACC_COLS].sum(axis=1)
    X["Children"] = X["Kidhome"] + X["Teenhome"]
    X["Age"] = 2014 - X["Year_Birth"]
    X["SpendPerPurchase"] = X[MNT_COLS].sum(axis=1) / X[PURCHASE_COLS].sum(axis=1).clip(lower=1)
    return X.drop(columns=ACC_COLS + ["Kidhome", "Teenhome", "Year_Birth"])

svc_new = Pipeline([("features", FunctionTransformer(add_features)),
                    ("svm", make_pipe(clone(grid_svc.best_estimator_["model"]), new_num))])
```
- `FunctionTransformer` pune o funcție oarecare ca pas în pipeline. Funcția nu învață nimic din date → fără leakage.
- `.clip(lower=1)` evită împărțirea la 0 (clienți cu 0 cumpărări).
- **Rezultat:** PR-AUC 0,606 → **0,625**, p = 0,090 → îmbunătățire la fiecare fold, dar nesemnificativă cu doar 5 fold-uri.

---

## 16. Rezumat

| Termen | Pe scurt | La noi |
|---|---|---|
| metrică principală | aleasă înainte | PR-AUC / MAE |
| margine | distanța frontieră – cele mai apropiate puncte | maximizată de SVM |
| vectori suport | punctele care definesc frontiera | 44% (SVC), 81% (SVR) |
| C | prețul unei încălcări | C = 100 la RBF |
| kernel / γ | frontieră curbă / cât de locală | RBF, γ = 0,001 (aproape liniar) |
| ε (SVR) | tub fără penalizare | 0,05 std |
| scalare X / y | obligatorie / obligatorie la SVR | fără: PR-AUC 0,31 / R² 0,01 |
| calibrare | scor → probabilitate | sigmoid, Brier 0,088 |
| prag | ales pe OOF | 0,19 (recall 0,75, precizie 0,44) |
| reziduuri | real − prezis | pâlnie → log pe țintă |
| test t corectat | diferență reală? | SVM vs logistică p = 0,85 |
| model degenerat | prezice mereu 0 | SVC fără class_weight: recall 0,22 |

### Întrebări de verificare

1. De ce PR-AUC și nu acuratețea?
2. Ce face C mare? Dar γ mare?
3. De ce SVR fără scalarea țintei are R² ≈ 0?
4. De ce pragul se alege pe predicții OOF?
5. SVM-ul RBF are PR-AUC 0,606, regresia logistică 0,601, p = 0,845. Pe care îl păstrăm?
6. Ce arată pâlnia din reziduuri?

**Răspunsuri:**
1. Clasa 1 are 15%: „mereu 0” dă 85% acuratețe. PR-AUC baseline = 0,15 și măsoară ordonarea clienților, adică exact cum se folosește modelul.
2. C mare = margine îngustă, încearcă să clasifice fiecare punct (overfitting). γ mare = influență foarte locală, „insule” (overfitting).
3. ε și C implicite sunt mici față de o țintă cu std ~600: aproape toate erorile sunt ieftine, iar modelul rămâne aproape constant.
4. Altfel l-am alege pe test (sau pe date pe care modelul le-a văzut) → scor optimist.
5. Regresia logistică: scor egal (diferența nu e semnificativă), mai simplă, mai rapidă, probabilități directe.
6. Eroarea crește cu ținta: clienții care cheltuie mult sunt subestimați → o transformare logaritmică a țintei poate ajuta.
