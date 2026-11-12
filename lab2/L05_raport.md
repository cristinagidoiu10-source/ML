# Raport L05 — Ansambluri, selecția atributelor și concluziile proiectului (T8 Customer Personality Analysis)

Notebook-ul `L05_Ansambluri_selectie_concluzii_v2.ipynb` are **toate TODO-urile necompletate**, deci nu am greșeli de semnalat în cod. Mai jos e soluția completă, pe TODO-uri, cu explicații și observații.

Legendă: ❌ necompletat (toate)

> Cifrele din observații le-am obținut rulând exact codul din acest raport (scikit-learn 1.9, imbalanced-learn 0.14) pe `clean.csv` corectat după raportul L02, după ce am rulat codul din raportul L04 (L05 încarcă modelele salvate acolo). Pe Colab pot ieși mici diferențe; copiază în notebook cifrele care îți ies ție.

**Ordinea celulelor:** exercițiul „Găsiți greșelile” de la început are nevoie de date. Pune celula lui de cod **după** celula cu split-ul (TODO 1).

---

## TODO 0 — Setul de date (celula 6) ❌

Celula are încă `"..."` peste tot.

### Cod corect
```python
# TODO: aceleasi valori ca in laboratoarele 2-4
MY_NAME = "Customer Personality Analysis"   # IDENTIC cu L02-L04
TARGET_CLF = "Response"                     # 1 = a acceptat oferta din ultima campanie
POS_CLASS = 1
TARGET_REG = "MntTotal"                     # suma cheltuita in 2 ani

MY_DIR = f"{DATA_DIR}/{MY_NAME}"
print(MY_NAME, "|", TARGET_CLF, "|", TARGET_REG, "| clean.csv exists:", os.path.exists(f"{MY_DIR}/clean.csv"))
```

---

## TODO 1 — Cele două probleme (celula 9) ❌

Aceleași split-uri ca în L04 (același cod), ca rezultatele să fie comparabile.

### Cod corect
```python
# TODO: aceleasi doua probleme si aceleasi split-uri ca in laboratorul 4
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

CLEAN_CSV = f"{MY_DIR}/clean.csv"
assert os.path.exists(CLEAN_CSV), f"Nu exista {CLEAN_CSV}: rulati intai capitolul 5 din laboratorul 2"
df = pd.read_csv(CLEAN_CSV)

MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
LEAKY_R = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]   # aceeasi perioada ca MntTotal

Xc = df.drop(columns=[TARGET_CLF, TARGET_REG])
yc = (df[TARGET_CLF] == POS_CLASS).astype(int)
Xr = df.drop(columns=[TARGET_REG, TARGET_CLF] + MNT_COLS + LEAKY_R)
yr = df[TARGET_REG]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=SEED)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)

num_cols_c = Xc_train.select_dtypes(include="number").columns.tolist()
cat_cols_c = Xc_train.select_dtypes(exclude="number").columns.tolist()
num_cols_r = Xr_train.select_dtypes(include="number").columns.tolist()
cat_cols_r = Xr_train.select_dtypes(exclude="number").columns.tolist()

cv_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_r = KFold(n_splits=5, shuffle=True, random_state=SEED)
print("classification:", Xc_train.shape, "| regression:", Xr_train.shape)
```

```markdown
**Justificare:** aceleași decizii ca în L04: la clasificare scot ținta și `MntTotal` (suma coloanelor `Mnt*`); la regresie scot ținta, `Mnt*` (ținta e suma lor), `Num*Purchases` (aceeași perioadă de 2 ani, leakage) și `Response`. Split 80/20, stratificat la clasificare; `StratifiedKFold` / `KFold` cu 5 fold-uri amestecate. Datele nu sunt o serie în timp.
```

---

## TODO 2 — Funcția de evaluare (celula 12) ❌

Aceeași funcție ca în L04, plus două lucruri: `make_pipe` acceptă un pas opțional de selecție a atributelor (capitolul 9), iar `compare` construiește tabelul pentru mai multe modele.

### Cod corect
```python
# TODO: functia de evaluare din L04 + o functie de comparatie
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

METRICS = {
    "classification": {"PR_AUC": "average_precision", "F1": "f1", "recall": "recall", "bal_acc": "balanced_accuracy"},
    "regression":     {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
}

def evaluate(model, X, y, task, cv):
    """CV pe metricile problemei; intoarce (randul de tabel, scorurile pe fold-uri ale metricii principale)."""
    scoring = METRICS[task]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    sign = {name: -1 if scorer.startswith("neg_") else 1 for name, scorer in scoring.items()}
    main = next(iter(scoring))
    folds = sign[main] * res[f"test_{main}"]
    row = {main: folds.mean(), f"{main}_std": folds.std(), f"{main}_train": sign[main] * res[f"train_{main}"].mean()}
    row |= {name: sign[name] * res[f"test_{name}"].mean() for name in scoring if name != main}
    row["fit_time_s"] = res["fit_time"].mean()
    return row, folds

def compare(models, X, y, task, cv):
    """Tabelul sortat dupa metrica principala + scorurile pe fold-uri ale fiecarui model."""
    rows, folds = [], {}
    for name, model in models.items():
        row, folds[name] = evaluate(model, X, y, task, cv)
        rows.append({"model": name, **row})
    main = next(iter(METRICS[task]))
    table = pd.DataFrame(rows).set_index("model").sort_values(main, ascending=(task == "regression"))
    return table, folds

EDU_ORDER = [["Basic", "Graduation", "Master", "PhD"]]

def make_pipe(model, num_cols, scaler=None, select=None):
    """prep (+ selectie optionala) + model; scaler="passthrough" -> fara scalare (arbori)."""
    num_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler() if scaler is None else scaler)])
    ord_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=EDU_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))])
    nom_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False))])
    prep = ColumnTransformer([("num", num_steps, num_cols), ("ord", ord_steps, ["Education"]),
                              ("nom", nom_steps, ["Marital_Status"])])
    steps = [("prep", prep)] + ([("select", select)] if select is not None else []) + [("model", model)]
    return Pipeline(steps)

# verificare pe modelele din L04 (salvate acolo) + modelele simple
from sklearn.linear_model import LogisticRegression, LinearRegression
svc_l04 = joblib.load(f"{MODELS_DIR}/L04_{MY_NAME}_svc.joblib")
svr_l04 = joblib.load(f"{MODELS_DIR}/L04_{MY_NAME}_svr.joblib")
ref_c, _ = compare({"logistic regression": make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), num_cols_c),
                    "SVM (L04)": svc_l04}, Xc_train, yc_train, "classification", cv_c)
ref_r, _ = compare({"linear regression": make_pipe(LinearRegression(), num_cols_r),
                    "SVR (L04)": svr_l04}, Xr_train, yr_train, "regression", cv_r)
display(ref_c.round(3))
display(ref_r.round(3))
```

**De ce `sparse_output=False`:** `SelectKBest` și ansamblurile primesc o matrice densă; cu 30 de coloane, memoria nu contează.

### Observații
```markdown
**Scoruri de referință (CV pe train):**
- **Clasificare (PR-AUC):** regresia logistică 0,601 ± 0,065; SVM din L04 0,606 ± 0,063 (practic egale). Baseline: 0,152.
- **Regresie (MAE):** regresia liniară 233,2 ± 12,1; SVR din L04 199,1 ± 10,9. Baseline: 527.
Acestea sunt scorurile pe care ansamblurile trebuie să le bată.
```

---

## Exercițiul „Găsiți greșelile” (celula 4) ❌

TODO-ul cere: copiați codul, corectați greșelile și comentați. Pe setul nostru, fără să atingem testul (testul se deschide abia la capitolul 10).

### Cod corect (celulă nouă, după TODO 1)
```python
# EXERCITIU: blocul corectat, pe setul nostru
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score, mean_absolute_error

# corectat: testul ramane inchis pana la capitolul 10 -> o parte din TRAIN joaca rolul de "test" aici
Xc_fit, Xc_hold, yc_fit, yc_hold = train_test_split(Xc_train, yc_train, test_size=0.2, stratify=yc_train, random_state=SEED)

# corectat: SelectKBest e un pas din pipeline (fit doar pe partea de antrenare a fiecarui fold), dupa imputare/encodare;
#           inainte invata din TOT setul -> leakage
# corectat: oob_score=True, altfel rf.oob_score_ nu exista (AttributeError)
# corectat: class_weight="balanced" - clasa 1 are 15%
rf_sel = make_pipe(RandomForestClassifier(n_estimators=500, oob_score=True, class_weight="balanced",
                                          min_samples_leaf=5, random_state=SEED, n_jobs=-1),
                   num_cols_c, scaler="passthrough", select=SelectKBest(f_classif, k=10))

# corectat: metrica de CLASIFICARE (PR-AUC), nu r2 (regresie); aceleasi fold-uri amestecate ca in rest
scores = cross_val_score(rf_sel, Xc_fit, yc_fit, cv=cv_c, scoring="average_precision")
print(f"CV PR-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

rf_sel.fit(Xc_fit, yc_fit)
print(f"OOB accuracy: {rf_sel['model'].oob_score_:.3f}")   # oob_score_ = acuratetea OOB (implicit), pe randurile nefolosite de fiecare arbore

# corectat: X_hold trece prin ACELASI pipeline (selectie inclusa); inainte X_test nu era transformat -> eroare de dimensiune
# corectat: PR-AUC / ROC-AUC se calculeaza din PROBABILITATI (predict_proba), nu din clasele prezise
p = rf_sel.predict_proba(Xc_hold)[:, 1]
print(f"holdout PR-AUC: {average_precision_score(yc_hold, p):.3f}")

# corectat: regresorul se antreneaza pe tinta de REGRESIE (MntTotal), nu pe y-ul de clasificare
# corectat: learning_rate=1.0 cu 3000 de arbori -> overfitting sigur; rata mica + oprire timpurie
gb = make_pipe(GradientBoostingRegressor(learning_rate=0.05, n_estimators=3000, n_iter_no_change=30,
                                         validation_fraction=0.15, random_state=SEED), num_cols_r, scaler="passthrough")
Xr_fit, Xr_hold, yr_fit, yr_hold = train_test_split(Xr_train, yr_train, test_size=0.2, random_state=SEED)
gb.fit(Xr_fit, yr_fit)
print(f"GB: stopped after {gb['model'].n_estimators_} trees | holdout MAE = {mean_absolute_error(yr_hold, gb.predict(Xr_hold)):.1f}")
```

### Observații
```markdown
**Greșelile găsite:**
1. `SelectKBest(k=10).fit(X, y)` pe **tot** setul, înainte de CV: selecția „vede” fold-urile de validare (leakage, demonstrat în L03 cu zgomot pur: 0,78 în loc de 0,50). Corect: pas în pipeline. **Fără eroare.**
2. `scoring="r2"` pentru un **clasificator**: R² e o metrică de regresie; pe 0/1 dă un număr fără sens. Corect: `average_precision` / `roc_auc`. **Fără eroare.**
3. `rf.oob_score_` fără `oob_score=True`: atributul nu există. **Eroare** (`AttributeError`).
4. `rf.predict(X_test)` pe date netransformate de selector: modelul a fost antrenat pe 10 coloane, `X_test` are toate coloanele. **Eroare** (număr diferit de atribute).
5. `roc_auc_score(y_test, p)` cu `p` = clasele prezise: AUC are nevoie de scoruri/probabilități; cu clase, curba ROC are un singur punct și AUC iese subestimat. **Fără eroare.**
6. `GradientBoostingRegressor(learning_rate=1.0, n_estimators=3000)`: fiecare arbore corectează complet eroarea celui dinainte; cu 3000 de arbori, modelul memorează train-ul (overfitting). Corect: rată mică (0,05–0,1) + oprire timpurie. **Fără eroare.**
7. Același `y` (de clasificare) e folosit și pentru regresor: problema greșită. **Fără eroare.**
8. `cv=5` fără amestecare și fără aceleași fold-uri ca restul modelelor. **Fără eroare.**
9. Folosirea lui `X_test` în timpul dezvoltării: testul se deschide o singură dată, la final. **Fără eroare.**
```

---

## TODO 3 — Votul majoritar (celula 17) ❌

### Cod corect
```python
# TODO: votul majoritar
from scipy.stats import binom

# 1. 5 modele independente, fiecare corect cu p = 0.7: votul e corect daca cel putin 3 din 5 sunt corecte
p1 = binom.sf(2, 5, 0.7)                     # P(X > 2) = P(X >= 3)
print(f"1. 5 independent models, p = 0.7 -> P(majority correct) = {p1:.3f}")

# 2. 3 dintre cele 5 sunt COPII ale aceluiasi model: cele 3 copii voteaza la fel si formeaza singure majoritatea
#    -> votul e corect exact cand modelul copiat e corect: P = 0.7 (celelalte 2 modele nu mai conteaza)
rng = np.random.default_rng(SEED)
n = 200_000
copy = rng.random(n) < 0.7                                   # modelul copiat (3 voturi identice)
others = (rng.random((n, 2)) < 0.7).sum(axis=1)              # 2 modele independente
p2 = ((3 * copy + others) >= 3).mean()
print(f"2. 3 copies + 2 independent models -> P(majority correct) = {p2:.3f} (simulated)")

# 3. p = 0.45 (mai slabe decat o moneda): ce se intampla cand M creste?
for M in [1, 5, 11, 21, 51, 101]:
    print(f"3. M = {M:3d}, p = 0.45 -> P(majority correct) = {binom.sf(M // 2, M, 0.45):.3f}")
```

### Observații
```markdown
**Observații:**
1. 5 modele independente cu p = 0,7: votul majoritar e corect cu probabilitatea **0,837**, mai mult decât un singur model (0,7).
2. Dacă 3 din 5 sunt copii ale aceluiași model, probabilitatea scade la **0,700**: cele 3 copii formează singure majoritatea, deci votul e corect exact când modelul copiat e corect. Ansamblul nu mai aduce nimic.
3. Cu p = 0,45, probabilitatea **scade** când M crește (0,45 → 0,41 → 0,37 → … → 0,16 la M = 101): votul amplifică și greșelile, nu doar reușitele.

**Pentru alegerea modelelor:** un ansamblu ajută doar dacă membrii sunt (1) **mai buni decât întâmplarea** și (2) **diferiți între ei** (greșesc pe exemple diferite). Mai multe copii ale aceluiași model, sau modele foarte corelate, nu ajută. De aceea random forest-ul alege atribute la întâmplare la fiecare împărțire, iar la voting/stacking combinăm modele de familii diferite (liniar, arbori, SVM).
```

---

## TODO 4 — Bagging, random forest, extra-trees (celula 22) ❌

### Cod corect
```python
# TODO: cat reduce media arborilor varianta fata de un singur arbore + max_features ales cu OOB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.metrics import average_precision_score, mean_absolute_error

N_TREES = 300
trees_c = {
    "one tree":    DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=SEED),
    "bagging":     BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced"),
                                     n_estimators=N_TREES, random_state=SEED, n_jobs=-1),
    "random forest": RandomForestClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "extra-trees": ExtraTreesClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
}
trees_r = {
    "one tree":    DecisionTreeRegressor(min_samples_leaf=5, random_state=SEED),
    "bagging":     BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5), n_estimators=N_TREES, random_state=SEED, n_jobs=-1),
    "random forest": RandomForestRegressor(N_TREES, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "extra-trees": ExtraTreesRegressor(N_TREES, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
}
table_bag_c, _ = compare({k: make_pipe(m, num_cols_c, scaler="passthrough") for k, m in trees_c.items()},
                         Xc_train, yc_train, "classification", cv_c)
table_bag_r, _ = compare({k: make_pipe(m, num_cols_r, scaler="passthrough") for k, m in trees_r.items()},
                         Xr_train, yr_train, "regression", cv_r)
display(table_bag_c[["PR_AUC", "PR_AUC_std", "PR_AUC_train", "F1"]].round(3))
display(table_bag_r[["MAE", "MAE_std", "MAE_train", "R2"]].round(3))

# max_features ales cu scorul OOB (fiecare rand e prezis doar de arborii care NU l-au vazut), comparat cu CV
rows = []
for mf in ["sqrt", 0.3, 0.5, 1.0]:
    rf_c = make_pipe(RandomForestClassifier(N_TREES, max_features=mf, min_samples_leaf=5, class_weight="balanced",
                                            oob_score=True, random_state=SEED, n_jobs=-1),
                     num_cols_c, scaler="passthrough").fit(Xc_train, yc_train)
    rf_r = make_pipe(RandomForestRegressor(N_TREES, max_features=mf, min_samples_leaf=5, oob_score=True,
                                           random_state=SEED, n_jobs=-1),
                     num_cols_r, scaler="passthrough").fit(Xr_train, yr_train)
    cv_pr, _ = evaluate(rf_c, Xc_train, yc_train, "classification", cv_c)
    cv_mae, _ = evaluate(rf_r, Xr_train, yr_train, "regression", cv_r)
    rows.append({"max_features": mf,
                 "OOB PR-AUC": average_precision_score(yc_train, rf_c["model"].oob_decision_function_[:, 1]),
                 "CV PR-AUC": cv_pr["PR_AUC"],
                 "OOB MAE": mean_absolute_error(yr_train, rf_r["model"].oob_prediction_),
                 "CV MAE": cv_mae["MAE"]})
display(pd.DataFrame(rows).set_index("max_features").round(3))
```

**De ce `min_samples_leaf=5` peste tot:** compar efectul **mediei**, nu al regularizării; toate variantele au aceleași frunze minime. `oob_decision_function_` / `oob_prediction_` sunt predicțiile OOB brute, ca să calculez PR-AUC și MAE, nu acuratețea implicită din `oob_score_`.

### Observații
```markdown
**Observații:**
- **Clasificare:** un singur arbore are PR-AUC 0,324; bagging 0,527, random forest 0,566, extra-trees 0,584. Media arborilor aproape dublează scorul. Diferența train – validare scade de la 0,57 (0,897 − 0,324) la ≈ 0,36–0,42, dar rămâne mare: arborii individuali memorează tot.
- **Regresie:** MAE scade de la 233,5 (un arbore) la 192 (bagging, random forest) și 187,3 (extra-trees), adică ≈ −45, de 4 ori abaterea standard. Arborele singur are aceeași eroare pe train (126) ca ansamblurile, dar pe validare greșește mult mai mult: media reduce **varianța**, nu bias-ul.
- **`max_features` (OOB):** la clasificare cel mai bun OOB e la **0,3** (0,599); la regresie la **0,5** (MAE OOB 186,2). Clasificarea preferă mai puține atribute pe împărțire (arbori mai diferiți), regresia mai multe (`Income` e de departe cel mai important atribut și trebuie să fie disponibil des).
- **OOB față de CV:** la regresie sunt foarte apropiate (diferențe de 3–4 unități). La clasificare OOB e cu 0,03–0,05 peste CV: modelul OOB e antrenat pe tot train-ul (fiecare rând e prezis de ≈ 1/3 din arbori, dar pădurea vede mai multe date decât un fold). Ordinea valorilor `max_features` se păstrează în mare, deci OOB e bun pentru o alegere rapidă, fără CV.
```

---

## TODO 5 — Boosting (celula 30) ❌

### Cod corect
```python
# TODO: boosting pe ambele probleme + curba "pierdere pe validare vs. numar de arbori"
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import log_loss

boost_c = {
    "random forest": make_pipe(RandomForestClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced",
                                                      random_state=SEED, n_jobs=-1), num_cols_c, scaler="passthrough"),
    "HistGB (early stop)": make_pipe(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, early_stopping=True,
                                                                    class_weight="balanced", random_state=SEED),
                                     num_cols_c, scaler="passthrough"),
    "GradientBoosting": make_pipe(GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=3,
                                                             subsample=0.8, random_state=SEED), num_cols_c, scaler="passthrough"),
}
boost_r = {
    "random forest": make_pipe(RandomForestRegressor(N_TREES, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
                               num_cols_r, scaler="passthrough"),
    "HistGB (early stop)": make_pipe(HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=1000,
                                                                   early_stopping=True, random_state=SEED),
                                     num_cols_r, scaler="passthrough"),
    "GradientBoosting": make_pipe(GradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, n_estimators=300,
                                                            max_depth=3, subsample=0.8, random_state=SEED),
                                  num_cols_r, scaler="passthrough"),
}
table_boost_c, folds_boost_c = compare(boost_c, Xc_train, yc_train, "classification", cv_c)
table_boost_r, folds_boost_r = compare(boost_r, Xr_train, yr_train, "regression", cv_r)
display(table_boost_c.round(3))
display(table_boost_r.round(3))

# curba: pierderea pe validare dupa fiecare arbore, pentru 3 rate de invatare (validare = 20% din train)
Xc_f, Xc_v, yc_f, yc_v = train_test_split(Xc_train, yc_train, test_size=0.2, stratify=yc_train, random_state=SEED)
Xr_f, Xr_v, yr_f, yr_v = train_test_split(Xr_train, yr_train, test_size=0.2, random_state=SEED)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for lr in [0.3, 0.1, 0.03]:
    gb_c = make_pipe(GradientBoostingClassifier(learning_rate=lr, n_estimators=1000, max_depth=3, random_state=SEED),
                     num_cols_c, scaler="passthrough").fit(Xc_f, yc_f)
    A_v = gb_c["prep"].transform(Xc_v)
    loss_c = [log_loss(yc_v, p[:, 1]) for p in gb_c["model"].staged_predict_proba(A_v)]
    gb_r = make_pipe(GradientBoostingRegressor(loss="absolute_error", learning_rate=lr, n_estimators=1000, max_depth=3,
                                               random_state=SEED), num_cols_r, scaler="passthrough").fit(Xr_f, yr_f)
    loss_r = [mean_absolute_error(yr_v, p) for p in gb_r["model"].staged_predict(gb_r["prep"].transform(Xr_v))]
    for ax, loss, name in [(axes[0], loss_c, "log-loss"), (axes[1], loss_r, "MAE")]:
        best = int(np.argmin(loss))
        ax.plot(range(1, len(loss) + 1), loss, label=f"lr = {lr}: min {loss[best]:.3f} at {best + 1} trees")
        ax.set(xlabel="number of trees", ylabel=f"validation {name}")
    print(f"lr = {lr}: classification min at {int(np.argmin(loss_c)) + 1} trees | regression min at {int(np.argmin(loss_r)) + 1} trees")
axes[0].set_title("Classification: validation log-loss")
axes[1].set_title("Regression: validation MAE")
for ax in axes:
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/L05_{MY_NAME}_boosting_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# ce a ales oprirea timpurie la HistGB (pe tot train-ul)
for name, X, y, table in [("classification", Xc_train, yc_train, boost_c), ("regression", Xr_train, yr_train, boost_r)]:
    hgb = table["HistGB (early stop)"].fit(X, y)
    print(f"HistGB {name}: early stopping chose {hgb['model'].n_iter_} trees (of max 1000)")
```

**De ce `loss="absolute_error"` la regresie:** metrica principală e MAE; ținta are clienți foarte mari (outlieri), iar pierderea absolută nu se lasă trasă de ei.

### Observații
```markdown
**Observații:**
- **Poziția minimului:** cu cât rata de învățare e mai mică, cu atât minimul vine mai târziu. La clasificare: 29 de arbori (lr = 0,3), 137 (0,1), 378 (0,03); la regresie 89 → 42 → 408 (la 0,3 curba e zgomotoasă). Rata × numărul de arbori rămâne aproximativ constantă. Rata mică are nevoie de mai mulți arbori, dar curba e mai netedă și minimul mai stabil.
- **Clasificare:** cel mai bun boosting e **GradientBoosting** (PR-AUC 0,598 ± 0,032), peste random forest (0,566 ± 0,040) cu 0,032, adică cam o abatere standard: nu e o diferență clară. **Dar** GradientBoosting nu are `class_weight`: recall 0,384, F1 0,475; la prag 0,5 ratează 60% dintre clienții interesați. HistGB (cu `class_weight`) are 0,553.
- **Regresie:** **GradientBoosting** (MAE 182,7 ± 9,7) și HistGB (185,5 ± 12,0) bat random forest-ul (192,2) cu 7–10, sub o abatere standard.
- **Cel mai rapid:** HistGB (≈ 0,4–0,5 s pe antrenare, față de ≈ 1,4 s GradientBoosting și ≈ 0,9 s random forest). Oprirea timpurie a ales doar 67 de arbori la clasificare și 104 la regresie (din maximum 1000).
```

---

## TODO 6 — Voting și stacking (celula 35) ❌

**Membrii:** trei familii diferite, ca greșelile să nu fie corelate (TODO 3): un model liniar, un SVM (din L04) și un ansamblu de arbori.

### Cod corect
```python
# TODO: voting si stacking din modele DIFERITE, pe ambele probleme
from sklearn.ensemble import VotingClassifier, StackingClassifier, VotingRegressor, StackingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeCV

members_c = {
    "logistic": make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), num_cols_c),
    "SVM (L04)": CalibratedClassifierCV(svc_l04, method="sigmoid", cv=3),          # SVC nu are predict_proba -> calibrat
    "HistGB": boost_c["HistGB (early stop)"],
}
members_r = {
    "linear": make_pipe(LinearRegression(), num_cols_r),
    "SVR (L04)": svr_l04,
    "random forest": boost_r["random forest"],
}
combos_c = {
    "voting (soft)": VotingClassifier(list(members_c.items()), voting="soft", n_jobs=-1),
    "stacking": StackingClassifier(list(members_c.items()), final_estimator=LogisticRegression(class_weight="balanced"),
                                   cv=5, n_jobs=-1),
}
combos_r = {
    "voting": VotingRegressor(list(members_r.items()), n_jobs=-1),
    "stacking": StackingRegressor(list(members_r.items()), final_estimator=RidgeCV(), cv=5, n_jobs=-1),
}
table_comb_c, folds_comb_c = compare(members_c | combos_c, Xc_train, yc_train, "classification", cv_c)
table_comb_r, folds_comb_r = compare(members_r | combos_r, Xr_train, yr_train, "regression", cv_r)
display(table_comb_c.round(3))
display(table_comb_r.round(3))

# pe ce membru se bazeaza meta-modelul? (ponderile lui, antrenat pe tot train-ul)
stack_c = combos_c["stacking"].fit(Xc_train, yc_train)
stack_r = combos_r["stacking"].fit(Xr_train, yr_train)
print("stacking (classification) weights:", dict(zip(members_c, stack_c.final_estimator_.coef_[0].round(3))))
print("stacking (regression) weights:    ", dict(zip(members_r, stack_r.final_estimator_.coef_.round(3))))
```

### Observații
```markdown
**Observații:**
- **Clasificare:** combinările bat cel mai bun membru: voting (soft) 0,630 ± 0,048, stacking 0,625 ± 0,044, față de SVM-ul calibrat 0,609 și regresia logistică 0,601. Câștigul (+0,02–0,03) e **sub** o abatere standard (≈ 0,05–0,06). La prag 0,5, voting-ul are recall doar 0,584; stacking-ul (meta-model cu `class_weight`) păstrează recall 0,788.
- **Regresie:** stacking-ul (MAE 186,1 ± 10,1) bate random forest-ul (192,2) cu 6, sub o abatere standard; voting-ul (193,0) nu bate cel mai bun membru: media simplă e trasă în jos de regresia liniară (233).
- **Timp:** stacking-ul e cel mai scump: 4,7 s la clasificare și 8,9 s la regresie pe antrenare (de 10–100 ori mai mult decât un membru), pentru că antrenează fiecare membru de 6 ori (5 fold-uri interne + final).
- **Pe ce se bazează stacking-ul:** la clasificare, pe **regresia logistică** (2,71) și **HistGB** (2,73) în mod egal, SVM-ul contează puțin (0,85): SVM-ul și regresia logistică spun aproape același lucru. La regresie, pe **random forest** (0,61) și SVR (0,37); regresia liniară primește 0,02, adică e practic ignorată.
```

---

## TODO 7 — Dezechilibrul și transformarea țintei (celula 41) ❌

### Cod corect
```python
# TODO (clasificare): strategia pentru dezechilibru, aleasa pe rezultate
from imblearn.ensemble import BalancedRandomForestClassifier

imb_c = {
    "RF, no weights": RandomForestClassifier(N_TREES, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "RF, class_weight=balanced": RandomForestClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced",
                                                        random_state=SEED, n_jobs=-1),
    "RF, balanced_subsample": RandomForestClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced_subsample",
                                                     random_state=SEED, n_jobs=-1),
    "BalancedRandomForest": BalancedRandomForestClassifier(N_TREES, sampling_strategy="all", replacement=True,
                                                           bootstrap=False, random_state=SEED, n_jobs=-1),
    "HistGB, no weights": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, early_stopping=True, random_state=SEED),
    "HistGB, class_weight=balanced": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, early_stopping=True,
                                                                    class_weight="balanced", random_state=SEED),
}
table_imb, _ = compare({k: make_pipe(m, num_cols_c, scaler="passthrough") for k, m in imb_c.items()},
                       Xc_train, yc_train, "classification", cv_c)
display(table_imb[["PR_AUC", "PR_AUC_std", "recall", "F1", "bal_acc"]].round(3))
```

```python
# TODO (regresie): are tinta o asimetrie care cere o transformare? Experiment: log1p vs. tinta originala
print(f"target skew: {yr_train.skew():.2f} | mean {yr_train.mean():.0f} vs median {yr_train.median():.0f}")

def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)   # log(1 + y): merge si la y = 0

log_r = {
    "linear": make_pipe(LinearRegression(), num_cols_r),
    "linear, log target": log_target(make_pipe(LinearRegression(), num_cols_r)),
    "random forest": boost_r["random forest"],
    "random forest, log target": log_target(boost_r["random forest"]),
    "HistGB": boost_r["HistGB (early stop)"],
    "HistGB, log target": log_target(make_pipe(HistGradientBoostingRegressor(learning_rate=0.05, max_iter=1000,
                                                                             early_stopping=True, random_state=SEED),
                                               num_cols_r, scaler="passthrough")),
}
table_log, _ = compare(log_r, Xr_train, yr_train, "regression", cv_r)
display(table_log.round(3))
```

**De ce `HistGB, log target` are `loss="squared_error"` (implicit):** pe scara logaritmică, eroarea pătratică e deja robustă (un client de 2000 nu mai e de 10 ori mai „departe” decât unul de 200, ci doar cu ≈ 2,3 unități log).

### Observații
```markdown
**Observații:**
- **Clasificare:** fără ponderi, random forest-ul are recall **0,160** (ratează 84% dintre clienții interesați), deși PR-AUC e aproape același (0,569): ordinea e bună, dar pragul 0,5 nu i se potrivește. `class_weight="balanced"` urcă recall-ul la 0,636 (F1 0,523), iar **`BalancedRandomForest`** (fiecare arbore vede un eșantion echilibrat) la **0,680** (F1 0,542, acuratețe echilibrată 0,766), cu cel mai mic PR-AUC (0,545). Precizia scade pe măsură ce crește recall-ul. Aleg **`class_weight="balanced"`**: același PR-AUC ca fără ponderi, recall de 4 ori mai mare, și e disponibil și la HistGB (0,588 recall față de 0,324). Oricum, pragul final îl aleg separat (capitolul 10).
- **Regresie:** ținta are asimetrie 0,87 (media 614, mediana 396). Logaritmul **nu ajută**: random forest 192,2 → 192,3 (egal), HistGB 185,5 → 200,4 (mai rău), iar regresia liniară devine catastrofală (MAE 394, R² −19). Arborii sunt insensibili la transformări monotone ale țintei. Pe scara log, modelele optimizează eroarea **relativă**, dar noi măsurăm eroarea absolută (MAE), dominată de clienții mari. La regresia liniară, `expm1` amplifică exponențial orice extrapolare. Asimetria e moderată (sub 1), deci nu transform ținta.
```

---

## TODO 8 — Tabelul complet și testul t (celula 47) ❌

### Cod corect
```python
# TODO: tabelul "un singur model vs. ansambluri" + testul t corectat intre primele doua
from sklearn.dummy import DummyClassifier, DummyRegressor

all_c = {
    "baseline":                  make_pipe(DummyClassifier(strategy="prior"), num_cols_c),
    "logistic regression (L03)": members_c["logistic"],
    "SVM (L04)":                 svc_l04,
    "random forest (bagging)":   boost_c["random forest"],
    "HistGB (boosting)":         boost_c["HistGB (early stop)"],
    "voting (soft)":             combos_c["voting (soft)"],
    "stacking":                  combos_c["stacking"],
}
all_r = {
    "baseline":                  make_pipe(DummyRegressor(), num_cols_r),
    "linear regression":         members_r["linear"],
    "SVR (L04)":                 svr_l04,
    "random forest (bagging)":   boost_r["random forest"],
    "HistGB (boosting)":         boost_r["HistGB (early stop)"],
    "voting":                    combos_r["voting"],
    "stacking":                  combos_r["stacking"],
}
table_c, folds_c = compare(all_c, Xc_train, yc_train, "classification", cv_c)
table_r, folds_r = compare(all_r, Xr_train, yr_train, "regression", cv_r)
display(table_c.round(3))
display(table_r.round(3))

def ttest_pair(folds, a, b, n_rows):
    k = len(folds[a])
    t, p_val = corrected_ttest(folds[a], folds[b], n_rows * (k - 1) / k, n_rows / k)
    print(f"{a} vs {b}: diff = {folds[a].mean() - folds[b].mean():+.3f}, t = {t:.2f}, p = {p_val:.3f} -> "
          f"{'significant' if p_val < 0.05 else 'NOT significant'}")
    return p_val

# primele doua modele din fiecare tabel
p_top_c = ttest_pair(folds_c, *table_c.index[:2], len(Xc_train))
p_top_r = ttest_pair(folds_r, *table_r.index[:2], len(Xr_train))
# merita complexitatea? cel mai bun model vs. un model mult mai simplu
ttest_pair(folds_c, table_c.index[0], "logistic regression (L03)", len(Xc_train))
ttest_pair(folds_r, table_r.index[0], "random forest (bagging)", len(Xr_train))

pd.concat([table_c.assign(task="classification"), table_r.assign(task="regression")]).to_csv(
    f"{RESULTS_DIR}/L05_{MY_NAME}_cv_results.csv")
```

### Observații
```markdown
**Observații:**
- **Ordinea (clasificare, PR-AUC):** voting 0,630 > stacking 0,625 > SVM 0,606 > regresia logistică 0,601 > random forest 0,566 > HistGB 0,553 > baseline 0,152. Ansamblurile de **arbori** sunt sub modelele liniare; doar **combinările** (care includ un model liniar) le bat.
- **Ordinea (regresie, MAE):** HistGB 185,5 ≈ stacking 186,1 < random forest 192,2 < voting 193,0 < SVR 199,1 < liniară 233,2. Aici ansamblurile de arbori bat clar modelele individuale (relația cu venitul e neliniară).
- **Testul t corectat:** voting vs stacking p = 0,381; HistGB vs stacking p = 0,849: **nesemnificative**. Și voting vs regresia logistică: diff +0,029, p = 0,275 → nesemnificativ. HistGB vs random forest: p = 0,081 → nesemnificativ.
- **Ce aleg:** **clasificare → regresia logistică**: același scor ca voting-ul (diferența nu e semnificativă), de 20 de ori mai rapidă, dă probabilități directe și ponderi explicabile. **Regresie → HistGB**: e primul, e de 20 de ori mai rapid decât stacking-ul la același scor și nu are nevoie de imputare.
- **Cel mai lent:** stacking-ul (5–9 s pe antrenare). Nu merită: câștigul față de modelele simple e sub zgomotul fold-urilor.
```

---

## TODO 9 — Reglarea celui mai promițător ansamblu (celula 53) ❌

**Ce reglez:** HistGB la ambele probleme. E cel mai rapid ansamblu și are cei mai mulți hiperparametri care contează. **Spațiul:** `learning_rate` pe scară logaritmică (0,01–0,3); complexitatea arborilor prin `max_leaf_nodes` (8–64) și `min_samples_leaf` (10–80: cu 1646 de rânduri, frunze sub 10 clienți înseamnă memorare); `l2_regularization` logaritmic (0,001–10). `max_iter` îl lasă oprirea timpurie. **Bugetul:** 30 de încercări × 5 fold-uri = 150 de antrenări per problemă, ≈ 1 minut.

### Cod corect
```python
# TODO: RandomizedSearchCV pentru HistGB, pe ambele probleme
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import loguniform, randint

space = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_leaf_nodes": randint(8, 65),
    "model__min_samples_leaf": randint(10, 81),
    "model__l2_regularization": loguniform(1e-3, 10),
}
searches = {}
for task, base, X, y, cv, scoring in [
        ("classification", boost_c["HistGB (early stop)"], Xc_train, yc_train, cv_c, "average_precision"),
        ("regression", boost_r["HistGB (early stop)"], Xr_train, yr_train, cv_r, "neg_mean_absolute_error")]:
    search = RandomizedSearchCV(base, space, n_iter=30, cv=cv, scoring=scoring, random_state=SEED,
                                n_jobs=-1, return_train_score=True)
    t0 = time.time()
    search.fit(X, y)
    searches[task] = search
    sign = -1 if scoring.startswith("neg_") else 1
    default = sign * cross_val_score(base, X, y, cv=cv, scoring=scoring)
    best_std = search.cv_results_["std_test_score"][search.best_index_]
    print(f"{task}: {time.time() - t0:.0f} s | default {default.mean():.3f} -> tuned {sign * search.best_score_:.3f} "
          f"(std {best_std:.3f})")
    print("   best:", {k.replace("model__", ""): round(v, 4) for k, v in search.best_params_.items()})
```

### Observații
```markdown
**Observații:**
- **Clasificare:** HistGB urcă de la 0,553 (setări implicite) la **0,603** (+0,050, peste abaterea standard de 0,032). Valori: `learning_rate` ≈ 0,047, `max_leaf_nodes` = 54, **`min_samples_leaf` = 71**, `l2_regularization` ≈ 0,03. Câștigul vine din frunzele mari (71 de clienți): arborii nu mai pot memora cazuri individuale. Chiar și reglat, HistGB doar ajunge regresia logistică (0,601).
- **Regresie:** 185,5 → **182,5** (−3,0), mult sub abaterea standard (11,3): practic niciun câștig. Valori: `learning_rate` ≈ 0,10, `max_leaf_nodes` = 16 (arbori mici), `min_samples_leaf` = 48, `l2_regularization` ≈ 2,75 (regularizare puternică). Setările implicite erau deja aproape optime, deci în modelul final păstrez HistGB cu setările implicite.
- **Timp:** 30 de încercări × 5 fold-uri: ≈ 20 s la clasificare și 15 s la regresie.
```

---

## TODO 10 — Selecția atributelor (celula 59) ❌

Două familii, ambele **în pipeline** (deci reînvățate în fiecare fold): **filtru** (`SelectKBest` cu informația mutuală, k ales prin CV) și **încorporată** (`SelectFromModel` cu un model L1 / un random forest, care păstrează atributele cu pondere / importanță peste medie).

### Cod corect
```python
# TODO: selectia atributelor, doua familii, corect in pipeline, pe ambele probleme
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from functools import partial

mi_c = partial(mutual_info_classif, random_state=SEED)
mi_r = partial(mutual_info_regression, random_state=SEED)
lr_c = LogisticRegression(max_iter=1000, class_weight="balanced")
rf_r = RandomForestRegressor(N_TREES, min_samples_leaf=5, random_state=SEED, n_jobs=-1)

# FILTRU: cate atribute pastram? k ales prin CV, pe aceleasi fold-uri
k_grid = {"select__k": [5, 10, 15, 20, "all"]}
filt_c = GridSearchCV(make_pipe(lr_c, num_cols_c, select=SelectKBest(mi_c)), k_grid, cv=cv_c,
                      scoring="average_precision", n_jobs=-1).fit(Xc_train, yc_train)
filt_r = GridSearchCV(make_pipe(rf_r, num_cols_r, scaler="passthrough", select=SelectKBest(mi_r)), k_grid, cv=cv_r,
                      scoring="neg_mean_absolute_error", n_jobs=-1).fit(Xr_train, yr_train)
for name, g, sign in [("classification", filt_c, 1), ("regression", filt_r, -1)]:
    scores = dict(zip(g.cv_results_["param_select__k"], (sign * g.cv_results_["mean_test_score"]).round(3)))
    print(f"filter, {name}: score by k = {scores} -> best k = {g.best_params_['select__k']}")

# INCORPORATA: L1 (clasificare) / importanta din random forest (regresie), prag = media
emb_c = make_pipe(lr_c, num_cols_c, select=SelectFromModel(
    LogisticRegression(penalty="l1", solver="liblinear", C=0.05, class_weight="balanced")))
emb_r = make_pipe(rf_r, num_cols_r, scaler="passthrough", select=SelectFromModel(
    RandomForestRegressor(100, min_samples_leaf=5, random_state=SEED, n_jobs=-1), threshold="mean"))

table_sel_c, _ = compare({"all features": make_pipe(lr_c, num_cols_c), "filter (best k)": filt_c.best_estimator_,
                          "embedded (L1)": emb_c}, Xc_train, yc_train, "classification", cv_c)
table_sel_r, _ = compare({"all features": boost_r["random forest"], "filter (best k)": filt_r.best_estimator_,
                          "embedded (RF importance)": emb_r}, Xr_train, yr_train, "regression", cv_r)
display(table_sel_c[["PR_AUC", "PR_AUC_std", "F1"]].round(3))
display(table_sel_r[["MAE", "MAE_std", "R2"]].round(3))

# ce atribute raman? (selectia antrenata pe tot train-ul)
for name, pipe, X, y in [("filter, classification", filt_c.best_estimator_, Xc_train, yc_train),
                         ("embedded, classification", emb_c, Xc_train, yc_train),
                         ("filter, regression", filt_r.best_estimator_, Xr_train, yr_train),
                         ("embedded, regression", emb_r, Xr_train, yr_train)]:
    pipe.fit(X, y)
    names = pipe["prep"].get_feature_names_out()
    kept = pipe["select"].get_support()
    print(f"\n{name}: keeps {kept.sum()} of {len(names)}\n   dropped: {[n.split('__')[1] for n in names[~kept]]}")
```

### Observații
```markdown
**Observații:**
- **Filtru (informația mutuală):** la clasificare cel mai bun e **k = toate** (PR-AUC 0,601); cu 5, 10, 15, 20 de atribute scade la 0,345 / 0,423 / 0,501 / 0,528: fiecare atribut eliminat costă. La regresie, k = 20 = toate atributele (MAE 192,2); cu 5 atribute, 204,1.
- **Încorporată:** la clasificare, L1 păstrează **18 din 30** și are PR-AUC 0,607 (practic egal cu 0,601, dar cu abatere mai mare, 0,080). Elimină `Income`, `Kidhome`, `MntWines`, `MntFruits`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`, `Complain` și câteva categorii `Marital_Status`. Are sens: venitul și cheltuielile pe categorii sunt puternic corelate între ele (L02), iar L1 păstrează un singur reprezentant (`MntMeatProducts`). La regresie, importanța din random forest cu pragul „media” păstrează **doar 2 din 20** (`Income` domină cu 80% din importanță), iar MAE crește la 212,9 (+21): pragul e prea agresiv.
- **Păstrez selecția în modelul final? Nu.** Niciuna nu îmbunătățește scorul semnificativ, iar la regresie strică. Setul are doar ≈ 25 de atribute, deci nu avem o problemă de dimensiune. Selecția L1 ar fi utilă doar pentru un model mai ușor de explicat (18 atribute în loc de 30) la același scor.
```

---

## TODO 11 — Testul final (celula 66) ❌

**Modelul final** îl aleg din tabelul de la TODO 8, după regula din laborator: dacă diferența față de un model mai simplu nu e semnificativă, îl aleg pe cel mai simplu. **Pragul** (clasificare) îl aleg pe predicții out-of-fold din train, cu același criteriu ca în L04 (precizia maximă cu recall ≥ 0,75), **înainte** de test.

### Cod corect
```python
# TODO: testul final, O SINGURA DATA, pentru fiecare problema
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, r2_score,
                             root_mean_squared_error, classification_report)
from sklearn.inspection import permutation_importance

FINAL_C = "logistic regression (L03)"
FINAL_R = "HistGB (boosting)"
final_c = clone(all_c[FINAL_C])
final_r = clone(all_r[FINAL_R])

# CLASIFICARE: pragul pe predictii out-of-fold din TRAIN (precizia maxima cu recall >= 0.75)
p_oof = cross_val_predict(final_c, Xc_train, yc_train, cv=cv_c, method="predict_proba")[:, 1]
thresholds = np.arange(0.05, 0.96, 0.01)
ok = [t for t in thresholds if recall_score(yc_train, p_oof >= t) >= 0.75]
THRESHOLD = max(ok, key=lambda t: precision_score(yc_train, p_oof >= t))
print(f"threshold chosen on train (OOF): {THRESHOLD:.2f}")

# antrenare pe tot train-ul, O SINGURA evaluare pe test
final_c.fit(Xc_train, yc_train)
final_r.fit(Xr_train, yr_train)
p_te = final_c.predict_proba(Xc_test)[:, 1]
pred_c = (p_te >= THRESHOLD).astype(int)
pred_r = final_r.predict(Xr_test)

test_c = {"PR_AUC": average_precision_score(yc_test, p_te), "ROC_AUC": roc_auc_score(yc_test, p_te),
          "F1": f1_score(yc_test, pred_c), "recall": recall_score(yc_test, pred_c), "precision": precision_score(yc_test, pred_c)}
test_r = {"MAE": mean_absolute_error(yr_test, pred_r), "RMSE": root_mean_squared_error(yr_test, pred_r),
          "R2": r2_score(yr_test, pred_r)}

# interval bootstrap 95% pentru metrica principala (reesantionam randurile din test, cu inlocuire)
rng = np.random.default_rng(SEED)
def bootstrap_ci(metric, y_true, y_score, n=2000):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    values = [metric(y_true[i], y_score[i]) for i in idx]
    return np.percentile(values, [2.5, 97.5])

ci_c = bootstrap_ci(average_precision_score, yc_test, p_te)
ci_r = bootstrap_ci(mean_absolute_error, yr_test, pred_r)
print(f"\nCLASSIFICATION ({FINAL_C}): CV PR-AUC {table_c.loc[FINAL_C, 'PR_AUC']:.3f} | "
      f"TEST PR-AUC {test_c['PR_AUC']:.3f} [95% CI {ci_c[0]:.3f} - {ci_c[1]:.3f}]")
print({k: round(v, 3) for k, v in test_c.items()})
print(classification_report(yc_test, pred_c, digits=3))
print(f"REGRESSION ({FINAL_R}): CV MAE {table_r.loc[FINAL_R, 'MAE']:.1f} | "
      f"TEST MAE {test_r['MAE']:.1f} [95% CI {ci_r[0]:.1f} - {ci_r[1]:.1f}]")
print({k: round(v, 3) for k, v in test_r.items()})

# lift: cati clienti care raspund gasim in primii 20% clienti ordonati dupa probabilitate, fata de unii luati la intamplare
order = np.argsort(-p_te)
top20 = order[: int(0.2 * len(order))]
captured = yc_test.to_numpy()[top20].sum() / yc_test.sum()
print(f"top 20% of customers: {captured:.0%} of all responders -> lift@20% = {captured / 0.2:.2f}")

# reziduuri (regresie)
resid = yr_test - pred_r
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(yr_test, pred_r, s=10, alpha=0.5)
axes[0].plot([0, yr_test.max()], [0, yr_test.max()], "k--", linewidth=1)
axes[0].set(xlabel=f"real {TARGET_REG}", ylabel="predicted", title="Test: predicted vs. real")
axes[1].scatter(pred_r, resid, s=10, alpha=0.5)
axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
axes[1].set(xlabel="predicted", ylabel="residual", title="Test: residuals")
plt.tight_layout()
plt.show()

# importanta prin permutare, pe TEST, pe coloanele originale (pipeline-ul primeste coloanele brute)
imp_c = permutation_importance(final_c, Xc_test, yc_test, scoring="average_precision", n_repeats=20, random_state=SEED, n_jobs=-1)
imp_r = permutation_importance(final_r, Xr_test, yr_test, scoring="neg_mean_absolute_error", n_repeats=20, random_state=SEED, n_jobs=-1)
for name, imp, X in [("classification", imp_c, Xc_test), ("regression", imp_r, Xr_test)]:
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(8)
    print(f"\npermutation importance ({name}):\n{top.round(3)}")

# MDI (importanta "interna" a arborilor), doar ca termen de comparatie: random forest pe regresie
rf_mdi = clone(boost_r["random forest"]).fit(Xr_train, yr_train)
mdi = pd.Series(rf_mdi["model"].feature_importances_, index=rf_mdi["prep"].get_feature_names_out())
print("\nMDI (random forest, regression):\n", mdi.sort_values(ascending=False).head(8).round(3))

joblib.dump(final_c, f"{MODELS_DIR}/L05_{MY_NAME}_final_clf.joblib")
joblib.dump(final_r, f"{MODELS_DIR}/L05_{MY_NAME}_final_reg.joblib")
pd.DataFrame([{"task": "classification", "model": FINAL_C, "threshold": THRESHOLD, **test_c,
               "CI_low": ci_c[0], "CI_high": ci_c[1]},
              {"task": "regression", "model": FINAL_R, **test_r, "CI_low": ci_r[0], "CI_high": ci_r[1]}]
             ).to_csv(f"{RESULTS_DIR}/L05_{MY_NAME}_test.csv", index=False)
```

### Observații
```markdown
**Observații:**
- **Clasificare (regresia logistică, prag 0,53):** test PR-AUC **0,744** [IC 95%: 0,633 – 0,857], față de CV 0,601. Testul iese **mai bun** decât CV, iar valoarea din CV e chiar sub intervalul bootstrap: testul pare „mai ușor” decât fold-urile din train. Cu 63 de clienți pozitivi în test, câțiva clienți ușor de recunoscut schimbă mult scorul (abaterea între fold-uri era 0,065). Nu am modificat nimic după test. Pe test: recall 0,857, precizie 0,535, F1 0,659, ROC-AUC 0,917.
- **Regresie (HistGB):** test MAE **160,0** [IC 95%: 144,7 – 176,0], față de CV 185,5, tot mai bun decât în CV. R² = 0,826. Reziduurile au aceeași pâlnie ca în L04: erorile cresc cu valoarea țintei.
- **Lift la 20%:** dacă contactăm doar primii 20% clienți (ordonați după probabilitatea dată de model), găsim **79%** dintre clienții care acceptă oferta: de **3,97 ori** mai mulți decât dacă am alege 20% clienți la întâmplare. Cu o cincime din costul campaniei, obținem patru cincimi din răspunsuri.
- **Importanța prin permutare (clasificare):** `Recency` (0,117), `AcceptedCmp5` (0,096), `MntMeatProducts` (0,091), `AcceptedCmp3`, `NumCatalogPurchases`, `Teenhome`, `Customer_Days`, `Marital_Status`. Are sens: clienții recenți, care au mai acceptat oferte și cheltuie pe carne (produse scumpe) răspund mai des.
- **Regresie:** `Income` domină (MAE crește cu 366 dacă îl amestecăm), apoi `Customer_Days` (37), `Kidhome` (22), `Teenhome` (17). **MDI** (random forest) dă aceeași ordine: `Income` 0,81, `Customer_Days` 0,07, `Teenhome`, `Kidhome`. Diferența: MDI dă importanță și lui `Recency` (0,016), care la permutare aproape nu contează. MDI favorizează atributele continue cu multe valori posibile de împărțire, chiar dacă împărțirile nu ajută pe date noi.
```

---

## Contribuția (celula 70) ❌

Setul de date are chiar costul și câștigul campaniei: coloanele constante `Z_CostContact = 3` și `Z_Revenue = 11` (pe care le-am scos la curățare pentru că sunt constante). Le folosesc pentru un **prag ales după profit**, nu după F1.

**Ipoteză:** pragul care maximizează profitul (11 câștig pentru fiecare client care acceptă, 3 cost pentru fiecare contact) e mai mic decât pragul ales după F1, pentru că un contact costă puțin față de câștig. Deci merită să contactăm mai mulți clienți decât ar spune F1.

### Cod corect
```python
# CONTRIBUTIA: pragul ales dupa PROFIT (Z_Revenue = 11 pe client care accepta, Z_CostContact = 3 pe contact)
REVENUE, COST = 11, 3

def profit(y_true, contacted):
    y_true, contacted = np.asarray(y_true), np.asarray(contacted)
    return REVENUE * (y_true & contacted).sum() - COST * contacted.sum()

# pragul ales pe predictiile out-of-fold din TRAIN (aceleasi ca la capitolul 10)
profits = [profit(yc_train, p_oof >= t) for t in thresholds]
T_PROFIT = thresholds[int(np.argmax(profits))]
f1s = [f1_score(yc_train, p_oof >= t) for t in thresholds]
T_F1 = thresholds[int(np.argmax(f1s))]

rows = []
for name, contacted_tr, contacted_te in [
        ("contact everyone", np.ones(len(yc_train), bool), np.ones(len(yc_test), bool)),
        (f"threshold max F1 ({T_F1:.2f})", p_oof >= T_F1, p_te >= T_F1),
        (f"threshold recall>=0.75 ({THRESHOLD:.2f})", p_oof >= THRESHOLD, p_te >= THRESHOLD),
        (f"threshold max profit ({T_PROFIT:.2f})", p_oof >= T_PROFIT, p_te >= T_PROFIT)]:
    rows.append({"strategy": name, "contacted (train)": contacted_tr.mean(),
                 "profit per customer (train, OOF)": profit(yc_train, contacted_tr) / len(yc_train),
                 "profit per customer (test)": profit(yc_test, contacted_te) / len(yc_test)})
display(pd.DataFrame(rows).set_index("strategy").round(3))

plt.figure(figsize=(7, 3.8))
plt.plot(thresholds, np.array(profits) / len(yc_train))
plt.axvline(T_PROFIT, color="k", linestyle="--", linewidth=1)
plt.gca().set(xlabel="threshold", ylabel="profit per customer (train, OOF)", title="Which threshold earns the most?")
plt.show()
```

Pragurile sunt alese **doar pe train**. Coloana „test” doar raportează rezultatul, nu decide nimic.

### Observații
```markdown
**Contribuția:**
- **Ipoteza:** pragul ales după profit e mai mic decât cel ales după F1, pentru că un contact costă 3 și un răspuns aduce 11.
- **Experimentul:** pe predicțiile out-of-fold din train (aceleași ca la capitolul 10), calculez profitul pentru fiecare prag: 11 × (clienți contactați care acceptă) − 3 × (clienți contactați). Compar cu „contactăm pe toată lumea”, pragul F1 maxim și pragul cu recall ≥ 0,75.
- **Rezultatul:** a contacta pe toată lumea **pierde** 1,33 pe client (doar 15% acceptă, iar 0,15 × 11 < 3). Pragul care maximizează profitul este **0,59**, **identic** cu pragul F1 maxim: 22% dintre clienți contactați, +0,52 pe client în train și +0,74 pe client în test. Pragul cu recall ≥ 0,75 (0,53) câștigă puțin mai puțin (+0,49 în train, +0,71 în test).
- **Concluzia:** ipoteza e **respinsă**: pragul după profit nu e mai mic decât cel după F1, ci identic. Explicația: cu `class_weight="balanced"`, probabilitățile modelului sunt „umflate” (modelul a fost antrenat ca și cum clasele ar fi 50/50), deci 0,59 pe scara modelului corespunde unei probabilități reale mult mai mici, aproape de pragul teoretic 3/11 ≈ 0,27. Lecția practică: modelul transformă o campanie în pierdere (−1,33 pe client) într-una profitabilă (+0,5–0,7 pe client).
```

---

## Concluzii (celula 74) ❌

```markdown
**Concluzii:**

**1. Problema.**
- **Clasificare:** prezic dacă un client acceptă oferta campaniei (`Response`), ca să contactăm doar clienții cu șanse reale; doar 15% acceptă, deci a contacta pe toată lumea aduce pierdere.
- **Regresie:** prezic suma cheltuită de un client în 2 ani (`MntTotal`), ca să știm cât valorează fiecare client.

**2. Tabelul final**

| | Clasificare (`TARGET_CLF` = `Response`) | Regresie (`TARGET_REG` = `MntTotal`) |
|---|---|---|
| metrica principală și de ce | PR-AUC: clasa rară (15%), contează ordinea clienților; F1/recall ca verificare | MAE: eroarea medie în unități monetare, ușor de explicat, nu e dominată de câțiva clienți foarte mari |
| baseline (CV) | 0,152 | 527,0 |
| cel mai bun model individual (CV, medie ± std) | SVM RBF 0,606 ± 0,063 (regresia logistică 0,601 ± 0,065) | SVR RBF 199,1 ± 10,9 |
| cel mai bun ansamblu (CV, medie ± std) | voting (soft) 0,630 ± 0,048 | HistGB 185,5 ± 12,0 |
| testul t corectat între primele două (p) | voting vs stacking: p = 0,381 (voting vs regresia logistică: p = 0,275) | HistGB vs stacking: p = 0,849 |
| **modelul final** și de ce l-am ales | **regresia logistică** (`class_weight="balanced"`): același scor ca voting-ul (diferență nesemnificativă), mai simplă, mai rapidă, explicabilă | **HistGB**: primul în tabel, rapid, fără imputare; stacking-ul are același scor cu un timp de 20 de ori mai mare |
| scorul pe test [interval bootstrap 95%] | PR-AUC 0,744 [0,633 – 0,857] | MAE 160,0 [144,7 – 176,0], R² 0,826 |
| pragul ales (clasificare) / eroarea tipică (regresie) | 0,53 (precizia maximă cu recall ≥ 0,75): recall 0,857, precizie 0,535 pe test | ≈ 160 pe client (media țintei ≈ 614) |
| **contribuția proprie** și efectul ei | prag după profit (11 / 3 din `Z_Revenue` / `Z_CostContact`): +0,52 pe client în train, +0,74 în test, față de −1,33 dacă contactăm pe toată lumea | (L04) atribute agregate pentru SVM: PR-AUC +0,020, p = 0,09 |

**3. Ce a contat cel mai mult:** (1) **tratarea dezechilibrului**: recall de la 0,43 la 0,79 (L03) și de la 0,16 la 0,64 la random forest (L05), la același PR-AUC; (2) **alegerea modelului** la regresie: de la MAE 233 (liniar) la 185 (HistGB); (3) **curățarea și evitarea leakage-ului**: `Num*Purchases` scoase din regresie (R² scade de la 0,85 la ≈ 0,8, dar modelul e onest). Reglarea hiperparametrilor a adus ≈ 0 la modelele liniare și la HistGB pe regresie (+0,05 doar la HistGB pe clasificare), iar selecția atributelor nu a ajutat.

**4. Ce atribute folosește modelul:** la clasificare `Recency`, campaniile anterioare acceptate (`AcceptedCmp5`, `AcceptedCmp3`), `MntMeatProducts`, `NumCatalogPurchases`, copiii adolescenți, vechimea și starea civilă. La regresie, aproape totul e `Income`, apoi vechimea și copiii. Are sens pentru un lanț de magazine.

**5. Limite:** la regresie modelul subestimează clienții care cheltuie mult și greșește la clienți cu venit mare care cheltuie puțin; la clasificare, aproape jumătate dintre clienții contactați nu acceptă. Testul are doar 63 de pozitivi, deci scorurile de test sunt incerte (interval ± 0,11 la PR-AUC). Setul are 2058 de clienți ai unui singur magazin, dintr-o singură perioadă (2012–2014), deci poate să nu fie reprezentativ pentru alte perioade. `AcceptedCmp*` din campaniile anterioare sunt atribute legitime doar dacă le avem înaintea noii campanii (le avem).

**6. Ce am face în continuare:** atribute agregate (total campanii acceptate, copii, vârstă), un prag ales după costul real al unui contact, validare încrucișată repetată (scorurile au abateri mari), date despre comportamentul recent (vizite, coș) în locul cheltuielilor pe 2 ani.

**7. Clasificare vs. regresie:** regresia a fost mai „ușoară” (R² ≈ 0,8, față de PR-AUC ≈ 0,6 la o clasă de 15%). Cheltuielile depind puternic de un singur atribut stabil (venitul), pe când decizia de a accepta o ofertă depinde de lucruri pe care nu le avem în date (oferta în sine, momentul, nevoia clientului).
```

---

## Celulele AI și tema (celula 75)

Celulele AI sunt goale: completează-le cinstit dacă ai folosit un asistent. Tema (fișa, partea a doua) nu e notată în cod; dacă ai completat fișa din L04, folosește-o.
