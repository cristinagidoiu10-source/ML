# Raport L06 (bonus) — Keras vs. scikit-learn (T8 Customer Personality Analysis)

Notebook-ul `L06_Bonus_Keras_v2.ipynb` are **toate sarcinile necompletate** (celula de configurare are încă `"..."`, deci `clean.csv exists: False`). Nu am greșeli de semnalat în cod. Mai jos e soluția completă, pe sarcini.

Legendă: ❌ necompletat (toate)

> **Ce am verificat și ce nu:** pe calculatorul tău nu am putut instala Keras (pachetele TensorFlow / JAX / PyTorch au căi de fișiere prea lungi pentru Windows fără „long paths”, iar asta e o setare de sistem pe care nu o schimb). Codul scikit-learn de aici e același ca în L04/L05, verificat acolo. **Codul Keras nu l-am rulat**: l-am scris după API-ul Keras 3 (cel din Colab, `Keras 3.13`), dar rulează-l celulă cu celulă și spune-mi dacă apare vreo eroare. Observațiile au locuri marcate cu **[…]** pentru cifrele care îți ies ție; textul din jur spune ce să cauți și cum să interpretezi.

**Nume:** exemplele (Penguins) folosesc `num_c`, `cat_c`, `prep_c`, `A_c_fit`, `build_model`, `keras_cv`. În codul tău folosesc alte nume, ca să nu strici celulele de exemplu care rulează după.

---

## Sarcina 0 — Setul de date (celula 6) ❌

### Cod corect
```python
# TODO: aceleasi valori ca in laboratoarele 2-5
MY_NAME = "Customer Personality Analysis"   # IDENTIC cu L02-L05
TARGET_CLF = "Response"
POS_CLASS = 1
TARGET_REG = "MntTotal"

MY_DIR = f"{DATA_DIR}/{MY_NAME}"
print(MY_NAME, "|", TARGET_CLF, "|", TARGET_REG, "| clean.csv exists:", os.path.exists(f"{MY_DIR}/clean.csv"))
```

---

## Sarcina 1 — Cele două probleme (celula 9) ❌

Identic cu TODO 1 din L05 (același split, aceleași fold-uri, ca să putem pune rețeaua în tabelul din L05).

### Cod corect
```python
# SARCINA: aceleasi doua probleme si aceleasi split-uri ca in laboratoarele 4-5
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

CLEAN_CSV = f"{MY_DIR}/clean.csv"
assert os.path.exists(CLEAN_CSV), f"Nu exista {CLEAN_CSV}: rulati intai capitolul 5 din laboratorul 2"
df = pd.read_csv(CLEAN_CSV)

MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
LEAKY_R = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]

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
**Justificare:** aceleași decizii ca în L04–L05: la clasificare scot ținta și `MntTotal`; la regresie scot ținta, `Mnt*` (ținta e suma lor), `Num*Purchases` (aceeași perioadă de 2 ani, leakage) și `Response`. Split 80/20 stratificat la clasificare, aleator la regresie; 5 fold-uri amestecate. Datele nu sunt o serie în timp.
```

---

## Sarcina 2 — Funcția de evaluare (celula 12) ❌

### Cod corect
```python
# SARCINA: functia de evaluare din L04 (pentru modelele scikit-learn de comparatie)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

METRICS = {
    "classification": {"PR_AUC": "average_precision", "F1": "f1", "recall": "recall"},
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

EDU_ORDER = [["Basic", "Graduation", "Master", "PhD"]]

def make_prep(num_cols):
    """Preprocesarea din L03: matrice DENSA de numere, fara NaN (Keras nu accepta matrici sparse sau text)."""
    num_steps = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    ord_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=EDU_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))])
    nom_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False))])
    return ColumnTransformer([("num", num_steps, num_cols), ("ord", ord_steps, ["Education"]),
                              ("nom", nom_steps, ["Marital_Status"])])

def make_pipe(model, num_cols):
    return Pipeline([("prep", make_prep(num_cols)), ("model", model)])
```

**Notă:** `Education` ordinal (0–3) nu e scalat. La o rețea e acceptabil (valori mici, între 0 și 3), dar dacă vrei să fie pe aceeași scară, adaugă un `StandardScaler()` după `OrdinalEncoder`.

---

## Sarcina 3 — Datele pentru Keras (celula 15) ❌

### Cod corect
```python
# SARCINA: matrice float32 fara NaN, preprocesarea invatata DOAR pe partea de antrenare, + set de validare din train
# 20% din TRAIN = set de validare pentru antrenarea retelei (EarlyStopping, alegeri); testul ramane inchis
Xc_fit, Xc_val, yc_fit, yc_val = train_test_split(Xc_train, yc_train, test_size=0.2, stratify=yc_train, random_state=SEED)
Xr_fit, Xr_val, yr_fit, yr_val = train_test_split(Xr_train, yr_train, test_size=0.2, random_state=SEED)

# CLASIFICARE: tinta 0/1
prep_clf = make_prep(num_cols_c).fit(Xc_fit)
Ac_fit = prep_clf.transform(Xc_fit).astype("float32")
Ac_val = prep_clf.transform(Xc_val).astype("float32")
tc_fit = yc_fit.to_numpy().astype("float32")
tc_val = yc_val.to_numpy().astype("float32")

# REGRESIE: X preprocesat + tinta scalata (media 0, abaterea 1), ambele invatate doar pe partea "fit"
prep_reg = make_prep(num_cols_r).fit(Xr_fit)
Ar_fit = prep_reg.transform(Xr_fit).astype("float32")
Ar_val = prep_reg.transform(Xr_val).astype("float32")
y_scaler = StandardScaler().fit(yr_fit.to_numpy().reshape(-1, 1))              # StandardScaler vrea o matrice cu o coloana
tr_fit = y_scaler.transform(yr_fit.to_numpy().reshape(-1, 1)).ravel().astype("float32")
tr_val = y_scaler.transform(yr_val.to_numpy().reshape(-1, 1)).ravel().astype("float32")

def to_money(t_scaled):
    """Predictiile retelei (tinta scalata) -> unitati monetare."""
    return y_scaler.inverse_transform(np.asarray(t_scaled).reshape(-1, 1)).ravel()

print("classification:", Ac_fit.shape, Ac_val.shape, "| NaN:", np.isnan(Ac_fit).sum())
print("regression:    ", Ar_fit.shape, Ar_val.shape, "| scaled target: mean", tr_fit.mean().round(2), "std", tr_fit.std().round(2))
print("columns (classification):", list(prep_clf.get_feature_names_out()))
```

### Observații
```markdown
- **Coloane:** clasificarea are **30** de coloane (23 numerice + `Education` ordinal + 6 coloane one-hot pentru `Marital_Status`, inclusiv „infrequent”), regresia **20** (13 numerice + 1 + 6). Verifică cu `Ac_fit.shape[1]` / `Ar_fit.shape[1]`.
- **De ce doar pe `X_fit` / `y_fit`:** mediana, media și abaterea standard sunt „învățate”. Dacă le calculăm și pe `X_val`, validarea influențează antrenarea (leakage), iar scorul pe validare (după care oprim antrenarea și alegem configurația) iese prea optimist. Ținta scalată (media 0, abaterea 1) ajută optimizatorul: pierderea pornește de la valori în jur de 1, nu de la sute de mii (MSE pe sume de bani), deci learning rate-ul implicit merge.
```

---

## Sarcina 4 — Prima rețea (celula 24) ❌

Sarcina cere **definiția scrisă de voi**, fără `build_model`. Arhitectura: 2 straturi ascunse (16 și 8 neuroni, ReLU). E mică, pentru că avem doar ≈ 1300 de rânduri de antrenare.

### Cod corect
```python
# SARCINA: definitia retelei pentru fiecare problema, scrisa de mana; curbe de invatare; numarul de parametri
keras.utils.set_random_seed(SEED)

# CLASIFICARE: iesire sigmoid = P(clasa 1); pierdere binary cross-entropy
net_clf = keras.Sequential([
    keras.Input(shape=(Ac_fit.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
], name="net_clf")
net_clf.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])

# REGRESIE: iesire liniara (orice numar real); pierdere MSE pe tinta scalata
net_reg = keras.Sequential([
    keras.Input(shape=(Ar_fit.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1),
], name="net_reg")
net_reg.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse",
                metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

hist_clf = net_clf.fit(Ac_fit, tc_fit, validation_data=(Ac_val, tc_val), epochs=200, batch_size=32, verbose=0)
hist_reg = net_reg.fit(Ar_fit, tr_fit, validation_data=(Ar_val, tr_val), epochs=200, batch_size=32, verbose=0)

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for row, (hist, metric, title) in enumerate([(hist_clf, "pr_auc", "classification"), (hist_reg, "mae", "regression")]):
    for col, key in enumerate(["loss", metric]):
        axes[row, col].plot(hist.history[key], label="train")
        axes[row, col].plot(hist.history[f"val_{key}"], label="validation")
        axes[row, col].set(xlabel="epoch", ylabel=key, title=f"{title} - {key}")
        axes[row, col].legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/L06_{MY_NAME}_first_nets.png", dpi=150, bbox_inches="tight")
plt.show()

for name, hist in [("classification", hist_clf), ("regression", hist_reg)]:
    val_loss = np.array(hist.history["val_loss"])
    print(f"{name}: lowest validation loss {val_loss.min():.4f} at epoch {val_loss.argmin() + 1} (of 200)")

# numarul de parametri, de mana: fiecare strat Dense are (intrari x neuroni) ponderi + neuroni bias-uri
def count_params(n_inputs, units):
    sizes = [n_inputs, *units]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

for name, net, A in [("classification", net_clf, Ac_fit), ("regression", net_reg, Ar_fit)]:
    by_hand = count_params(A.shape[1], [16, 8, 1])
    print(f"{name}: {by_hand} parameters by hand | Keras: {net.count_params()} | training rows: {len(A)}")
net_clf.summary()
```

**Calculul de mână (clasificare, 30 de intrări):** 30·16 + 16 = 496; 16·8 + 8 = 136; 8·1 + 1 = 9; total **641**. Regresie (20 de intrări): 20·16 + 16 = 336; + 136 + 9 = **481**.

### Observații
```markdown
- **Curbele:** pierderea pe train scade tot timpul. Pierderea pe validare scade în primele […] epoci (clasificare) / […] epoci (regresie), apoi stagnează și începe să crească: de acolo modelul învață zgomotul din train (overfitting). Curbele se depărtează după minimul pierderii pe validare.
- **Parametri:** 641 (clasificare) și 481 (regresie), față de 1316 rânduri de antrenare: un raport de ≈ 1 parametru la 2 rânduri, deci rețeaua poate memora destul de ușor un set atât de mic. De aceea e nevoie de oprire timpurie / regularizare (sarcina 5).
```

---

## Sarcina 5 — Overfitting provocat și reparat (celula 29) ❌

### Cod corect
```python
# SARCINA: overfitting intentionat (retea mare, fara regularizare), apoi reparat cu EarlyStopping, Dropout, L2
from keras import regularizers, callbacks
from sklearn.metrics import average_precision_score, mean_absolute_error

def make_net(n_inputs, task, units=(16, 8), dropout=0.0, l2=0.0, lr=1e-3, loss=None):
    """Aceeasi constructie ca la sarcina 4, cu regularizare optionala."""
    reg = regularizers.l2(l2) if l2 > 0 else None
    net = keras.Sequential([keras.Input(shape=(n_inputs,))])
    for u in units:
        net.add(layers.Dense(u, activation="relu", kernel_regularizer=reg))
        if dropout > 0:
            net.add(layers.Dropout(dropout))
    if task == "classification":
        net.add(layers.Dense(1, activation="sigmoid"))
        net.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss or "binary_crossentropy")
    else:
        net.add(layers.Dense(1))
        net.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss or "mse")
    return net

def early_stop():
    return callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

BIG = (256, 256, 128)
variants = {
    "big, no regularization":    dict(units=BIG, stop=False),
    "big + EarlyStopping":       dict(units=BIG, stop=True),
    "big + Dropout 0.3 + ES":    dict(units=BIG, dropout=0.3, stop=True),
    "big + L2 1e-3 + ES":        dict(units=BIG, l2=1e-3, stop=True),
}
data = {"classification": (Ac_fit, tc_fit, Ac_val, tc_val), "regression": (Ar_fit, tr_fit, Ar_val, tr_val)}
rows, histories = [], {}
for task, (A_f, t_f, A_v, t_v) in data.items():
    for name, cfg in variants.items():
        cfg = dict(cfg)
        stop = cfg.pop("stop")
        keras.utils.set_random_seed(SEED)
        net = make_net(A_f.shape[1], task, **cfg)
        h = net.fit(A_f, t_f, validation_data=(A_v, t_v), epochs=300, batch_size=32, verbose=0,
                    callbacks=[early_stop()] if stop else [])
        histories[(task, name)] = h
        p = net.predict(A_v, verbose=0).ravel()
        score = average_precision_score(t_v, p) if task == "classification" else mean_absolute_error(yr_val, to_money(p))
        rows.append({"task": task, "variant": name, "epochs run": len(h.history["loss"]),
                     "validation PR-AUC / MAE": score, "params": net.count_params()})
display(pd.DataFrame(rows).set_index(["task", "variant"]).round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, task in zip(axes, data):
    for name in variants:
        ax.plot(histories[(task, name)].history["val_loss"], label=name)
    ax.set(xlabel="epoch", ylabel="validation loss", title=f"{task}: validation loss", yscale="log")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/L06_{MY_NAME}_regularization.png", dpi=150, bbox_inches="tight")
plt.show()
```

**De ce MAE în unități monetare pentru regresie:** pierderea e calculată pe ținta scalată; `to_money` transformă predicțiile înapoi, ca să compar direct cu MAE-ul din L05 (≈ 185).

### Observații
```markdown
- **Rețeaua mare fără regularizare** (≈ 107.000 de parametri la ≈ 1300 de rânduri) rulează toate cele 300 de epoci; pierderea pe validare crește mult după primele epoci: overfitting clar.
- **Cea mai bună variantă:** clasificare […] (PR-AUC […]); regresie […] (MAE […]). De obicei, la seturi atât de mici, EarlyStopping + Dropout sau EarlyStopping + L2 bat EarlyStopping singur.
- **Epoci cu EarlyStopping:** […] (clasificare) și […] (regresie); `restore_best_weights=True` revine la epoca cu cea mai mică pierdere pe validare (cu 20 de epoci înainte de oprire).
- **Comparație cu HistGB (L05):** oprirea timpurie a ales acolo 67 de arbori la clasificare și 104 la regresie. Ideea e aceeași: oprim când pierderea pe date nevăzute nu mai scade; la HistGB „pasul” e un arbore, la rețea o epocă.
```

---

## Sarcina 6 — Dezechilibru și pierderea la regresie (celula 35) ❌

### Cod corect
```python
# SARCINA (clasificare): class_weight si efectul asupra recall-ului si preciziei (prag 0.5, pe validare)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

def balanced_weights(y):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.asarray(y).astype(int))
    return {0: w[0], 1: w[1]}                                  # Keras vrea un dictionar {clasa: pondere}

rows = []
for name, cw in [("no class_weight", None), ("class_weight balanced", balanced_weights(tc_fit))]:
    keras.utils.set_random_seed(SEED)
    net = make_net(Ac_fit.shape[1], "classification", units=(32, 16), dropout=0.3)
    net.fit(Ac_fit, tc_fit, validation_data=(Ac_val, tc_val), epochs=300, batch_size=32, class_weight=cw,
            callbacks=[early_stop()], verbose=0)
    p = net.predict(Ac_val, verbose=0).ravel()
    rows.append({"variant": name, "PR-AUC": average_precision_score(tc_val, p), "recall": recall_score(tc_val, p >= 0.5),
                 "precision": precision_score(tc_val, p >= 0.5, zero_division=0), "F1": f1_score(tc_val, p >= 0.5)})
display(pd.DataFrame(rows).set_index("variant").round(3))
print("class weights:", {k: round(v, 2) for k, v in balanced_weights(tc_fit).items()})
```

```python
# SARCINA (regresie): MSE vs. MAE vs. Huber, pe tinta curata si pe una cu 5% erori simulate (doar in antrenare)
rng = np.random.default_rng(SEED)
corrupt = rng.random(len(tr_fit)) < 0.05
tr_fit_noisy = tr_fit.copy()
tr_fit_noisy[corrupt] = tr_fit_noisy[corrupt] + rng.normal(5, 1, corrupt.sum())    # +5 abateri standard: erori grosolane

rows = []
for loss_name, loss in [("MSE", "mse"), ("MAE", "mae"), ("Huber", keras.losses.Huber(delta=1.0))]:
    for data_name, target in [("clean", tr_fit), ("5% errors", tr_fit_noisy)]:
        keras.utils.set_random_seed(SEED)
        net = make_net(Ar_fit.shape[1], "regression", units=(32, 16), dropout=0.1, loss=loss)
        net.fit(Ar_fit, target, validation_data=(Ar_val, tr_val), epochs=300, batch_size=32,
                callbacks=[early_stop()], verbose=0)
        pred = to_money(net.predict(Ar_val, verbose=0))
        rows.append({"loss": loss_name, "training target": data_name, "validation MAE": mean_absolute_error(yr_val, pred)})
display(pd.DataFrame(rows).pivot(index="loss", columns="training target", values="validation MAE").round(1))
```

Validarea folosește **ținta curată** (`tr_val`): vrem să știm cât de bine prezice modelul realitatea, nu erorile.

### Observații
```markdown
- **Clasificare:** cu `class_weight` (≈ 0,59 pentru clasa 0 și ≈ 3,3 pentru clasa 1), recall-ul pe validare crește de la […] la […], iar precizia scade de la […] la […]. PR-AUC rămâne aproape același (ponderile schimbă pragul efectiv, nu ordinea clienților). Același efect ca în L03 la regresia logistică (recall 0,43 → 0,79).
- **Regresie:** pe datele curate cea mai bună pierdere e […]; cu 5% erori, MSE se strică cel mai mult (eroarea la pătrat dă o greutate uriașă celor 5% de valori greșite), iar MAE și Huber rămân aproape la fel. **Aleg MAE (sau Huber):** metrica noastră e MAE, iar ținta are clienți foarte mari (asimetrie 0,87), care la MSE ar trage modelul după ei.
```

---

## Sarcina 7 — Raport cu `sklearn.metrics` și comparație cu `MLP` (celula 41) ❌

### Cod corect
```python
# SARCINA: raportul pentru retea (clasificare: prag ales pe VALIDARE; regresie: unitati monetare) + MLP din scikit-learn
from sklearn.metrics import classification_report, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.compose import TransformedTargetRegressor

UNITS = (32, 16)

# CLASIFICARE: reteaua finala a sarcinilor 5-6 (Dropout + class_weight + EarlyStopping)
keras.utils.set_random_seed(SEED)
net_c = make_net(Ac_fit.shape[1], "classification", units=UNITS, dropout=0.3)
net_c.fit(Ac_fit, tc_fit, validation_data=(Ac_val, tc_val), epochs=300, batch_size=32,
          class_weight=balanced_weights(tc_fit), callbacks=[early_stop()], verbose=0)
p_val = net_c.predict(Ac_val, verbose=0).ravel()

# pragul: precizia maxima cu recall >= 0.75 (acelasi criteriu ca in L04-L05)
thresholds = np.arange(0.05, 0.96, 0.01)
ok = [t for t in thresholds if recall_score(tc_val, p_val >= t) >= 0.75]
threshold_net = max(ok, key=lambda t: precision_score(tc_val, p_val >= t))
print(f"threshold chosen on validation: {threshold_net:.2f}")
print(classification_report(tc_val, (p_val >= threshold_net).astype(int), digits=3))
# atentie: pragul e ales pe aceleasi date pe care il raportam -> raportul e usor optimist (testul ramane pentru final)

# MLPClassifier cu aceeasi arhitectura (fara class_weight: scikit-learn nu il are la MLP)
mlp_c = make_pipe(MLPClassifier(hidden_layer_sizes=UNITS, activation="relu", alpha=1e-3, early_stopping=True,
                                max_iter=500, random_state=SEED), num_cols_c).fit(Xc_fit, yc_fit)
print(f"PR-AUC on validation: Keras {average_precision_score(tc_val, p_val):.3f} | "
      f"MLPClassifier {average_precision_score(yc_val, mlp_c.predict_proba(Xc_val)[:, 1]):.3f}")

# REGRESIE: reteaua cu pierdere MAE, predictiile in unitati monetare
keras.utils.set_random_seed(SEED)
net_r = make_net(Ar_fit.shape[1], "regression", units=UNITS, dropout=0.1, loss="mae")
net_r.fit(Ar_fit, tr_fit, validation_data=(Ar_val, tr_val), epochs=300, batch_size=32, callbacks=[early_stop()], verbose=0)
pred_val = to_money(net_r.predict(Ar_val, verbose=0))
mlp_r = TransformedTargetRegressor(make_pipe(MLPRegressor(hidden_layer_sizes=UNITS, activation="relu", alpha=1e-3,
                                                          early_stopping=True, max_iter=500, random_state=SEED), num_cols_r),
                                   transformer=StandardScaler()).fit(Xr_fit, yr_fit)
pred_mlp = mlp_r.predict(Xr_val)
for name, pred in [("Keras", pred_val), ("MLPRegressor", pred_mlp)]:
    print(f"{name:12s}: MAE {mean_absolute_error(yr_val, pred):.1f} | RMSE {root_mean_squared_error(yr_val, pred):.1f} | "
          f"R2 {r2_score(yr_val, pred):.3f}")

# unde greseste reteaua de regresie? MAE pe sferturi ale tintei
quart = pd.qcut(yr_val, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
display(pd.DataFrame({"abs error": np.abs(yr_val - pred_val), "residual": yr_val - pred_val})
        .groupby(quart, observed=True).mean().round(1))
```

**Arhitectura „aceeași”:** 2 straturi ascunse (32, 16), ReLU, Adam, regularizare L2 (`alpha` la MLP), oprire timpurie. Diferențe care rămân: MLP-ul nu are Dropout și nici `class_weight`, iar oprirea lui timpurie folosește 10% din datele primite.

### Observații
```markdown
- **Keras vs. MLP:** PR-AUC Keras […] vs. MLPClassifier […]; MAE Keras […] vs. MLPRegressor […]. Cu aceeași arhitectură, diferențele sunt de obicei mici. Dacă Keras e mai bun la recall, e din cauza `class_weight` (MLPClassifier nu îl are).
- **Pragul:** […] (precizia maximă cu recall ≥ 0,75): găsim cel puțin 3 din 4 clienți interesați. Pragul e ales pe setul de validare, deci raportul pe validare e ușor optimist.
- **Unde greșește regresia:** ca la SVR (L04) și HistGB (L05), eroarea crește cu valoarea țintei: MAE mic la clienții mici (Q1), mare la cei mari (Q4), unde modelul **subestimează** (reziduul mediu pozitiv).
```

---

## Sarcina 8 — Validarea încrucișată a rețelei + tabelul din L05 (celula 47) ❌

Bucla e scrisă de mine (nu `keras_cv` din exemplu): în fiecare fold, preprocesarea și scalarea țintei se învață doar pe partea de antrenare, rețeaua e nouă, iar 15% din partea de antrenare e folosit pentru `EarlyStopping`.

### Cod corect
```python
# SARCINA: validarea incrucisata a retelei, aceleasi fold-uri ca in L05, + tabel + testul t corectat
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

def net_cv(X, y, task, cv, num_cols, units=UNITS, dropout=0.3, l2=0.0, lr=1e-3, loss=None, epochs=300, seed=SEED):
    rows = []
    y = np.asarray(y)
    for idx_fit, idx_val in cv.split(X, y):
        X_f, X_v, y_f, y_v = X.iloc[idx_fit], X.iloc[idx_val], y[idx_fit], y[idx_val]
        X_f, X_es, y_f, y_es = train_test_split(X_f, y_f, test_size=0.15, random_state=seed,
                                                stratify=y_f if task == "classification" else None)
        prep = make_prep(num_cols).fit(X_f)                               # doar pe antrenarea fold-ului
        A_f, A_es, A_v = (prep.transform(d).astype("float32") for d in (X_f, X_es, X_v))
        cw = None
        if task == "regression":
            ys = StandardScaler().fit(y_f.reshape(-1, 1))                 # si tinta, doar pe antrenarea fold-ului
            t_f, t_es = (ys.transform(d.reshape(-1, 1)).ravel() for d in (y_f, y_es))
        else:
            t_f, t_es, cw = y_f, y_es, balanced_weights(y_f)
        keras.backend.clear_session()                                    # sterge retelele din fold-urile anterioare
        keras.utils.set_random_seed(seed)
        net = make_net(A_f.shape[1], task, units=units, dropout=dropout, l2=l2, lr=lr, loss=loss)   # retea NOUA
        net.fit(A_f, t_f, validation_data=(A_es, t_es), epochs=epochs, batch_size=32, class_weight=cw,
                callbacks=[early_stop()], verbose=0)
        p = net.predict(A_v, verbose=0).ravel()
        if task == "regression":
            pred = ys.inverse_transform(p.reshape(-1, 1)).ravel()
            rows.append({"MAE": mean_absolute_error(y_v, pred), "RMSE": root_mean_squared_error(y_v, pred),
                         "R2": r2_score(y_v, pred)})
        else:
            rows.append({"PR_AUC": average_precision_score(y_v, p), "F1": f1_score(y_v, p >= 0.5),
                         "recall": recall_score(y_v, p >= 0.5)})
    folds = pd.DataFrame(rows)
    main = folds.columns[0]
    return {main: folds[main].mean(), f"{main}_std": folds[main].std(ddof=0), **folds.iloc[:, 1:].mean()}, folds[main].to_numpy()

t0 = time.time()
row_nc, folds_nc = net_cv(Xc_train, yc_train, "classification", cv_c, num_cols_c, dropout=0.3)
time_net_c = time.time() - t0
t0 = time.time()
row_nr, folds_nr = net_cv(Xr_train, yr_train, "regression", cv_r, num_cols_r, dropout=0.1, loss="mae")
time_net_r = time.time() - t0

# modelele finale din L05, pe ACELEASI fold-uri, cu aceeasi functie de evaluare
best_c = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), num_cols_c)
best_r = make_pipe(HistGradientBoostingRegressor(learning_rate=0.05, max_iter=1000, early_stopping=True,
                                                 loss="absolute_error", random_state=SEED), num_cols_r)
row_bc, folds_bc = evaluate(best_c, Xc_train, yc_train, "classification", cv_c)
row_br, folds_br = evaluate(best_r, Xr_train, yr_train, "regression", cv_r)

table_net = pd.DataFrame([{"task": "classification", "model": "Keras network", **row_nc, "cv time (s)": time_net_c},
                          {"task": "classification", "model": "logistic regression (L05 final)", **row_bc},
                          {"task": "regression", "model": "Keras network", **row_nr, "cv time (s)": time_net_r},
                          {"task": "regression", "model": "HistGB (L05 final)", **row_br}]).set_index(["task", "model"])
display(table_net.round(3))

# testul t corectat: reteaua vs. cel mai bun model din L05
for task, f_net, f_best, n in [("classification", folds_nc, folds_bc, len(Xc_train)), ("regression", folds_nr, folds_br, len(Xr_train))]:
    t, p_val = corrected_ttest(f_net, f_best, n * 4 / 5, n / 5)
    print(f"{task}: network - best = {f_net.mean() - f_best.mean():+.3f}, p = {p_val:.3f} -> "
          f"{'significant' if p_val < 0.05 else 'NOT significant'}")
```

`corrected_ttest` vine din celula cu funcțiile ajutătoare a exemplului (e permisă).

### Observații
```markdown
- **Locul rețelei:** clasificare PR-AUC […] ± […] față de regresia logistică 0,601 ± 0,065; regresie MAE […] ± […] față de HistGB 185,5 ± 12,0. Testul t corectat: p = […] / […].
- **Timp:** validarea încrucișată a rețelei a durat […] s / […] s, față de ≈ 1 s pentru regresia logistică și ≈ 2 s pentru HistGB (5 fold-uri).
- **Aș alege rețeaua?** Pentru date tabelare ca ale noastre (≈ 1650 de rânduri, 25 de coloane), **nu**: în cel mai bun caz egalează modelele din L05 (diferență nesemnificativă), e de zeci de ori mai lentă, depinde de seed (sarcina 10) și e mai greu de explicat. Rețelele câștigă la date mari și nestructurate (imagini, text, sunet); pe tabele mici, arborii cu boosting și modelele liniare rămân alegerea standard.
```

---

## Sarcina 9 — Căutarea hiperparametrilor (celula 54) ❌

**Spațiul:** arhitectura (un strat de 16 sau două straturi 32-16), Dropout (0 sau 0,3) și rata de învățare (0,001 sau 0,003): 2 × 2 × 2 = **8 configurații**. Rețeaua mare (64-32) am scos-o: în sarcina 5 rețelele mari doar făceau overfitting. **Bugetul:** 3 fold-uri în loc de 5 și maximum 150 de epoci, adică 8 × 3 = 24 de antrenări per problemă. Pe CPU-ul din Colab, o antrenare durează ≈ 10–30 s, deci căutarea durează ≈ 5–10 minute per problemă.

### Cod corect
```python
# SARCINA: cautarea hiperparametrilor retelei (grila mica, evaluata cu bucla proprie pe 3 fold-uri)
from itertools import product

cv3_c = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv3_r = KFold(n_splits=3, shuffle=True, random_state=SEED)
grid_net = {"units": [(16,), (32, 16)], "dropout": [0.0, 0.3], "lr": [1e-3, 3e-3]}   # 2 x 2 x 2 = 8 configuratii
EPOCHS_SEARCH = 150                       # plafon mai mic la cautare; EarlyStopping opreste oricum mai devreme

problems = {"classification": (Xc_train, yc_train, cv3_c, num_cols_c, None, "PR_AUC"),
            "regression":     (Xr_train, yr_train, cv3_r, num_cols_r, "mae", "MAE")}
results_net, search_time = {}, {}
for task, (X, y, cv, num_cols, loss, main) in problems.items():
    rows = []
    t0 = time.time()
    for units, dropout, lr in product(*grid_net.values()):
        row, _ = net_cv(X, y, task, cv, num_cols, units=units, dropout=dropout, lr=lr, loss=loss, epochs=EPOCHS_SEARCH)
        rows.append({"units": units, "dropout": dropout, "lr": lr, **row})        # units ramane tuplu (nu text)
        print(f"  {task}: units={units}, dropout={dropout}, lr={lr} -> {main} = {row[main]:.3f}")
    search_time[task] = time.time() - t0
    results_net[task] = (pd.DataFrame(rows)
                         .sort_values(main, ascending=(task == "regression"))   # MAE: mai mic = mai bine
                         .reset_index(drop=True))
    print(f"{task}: {len(rows)} configurations x {cv.get_n_splits()} folds in {search_time[task]:.0f} s")
    display(results_net[task].head(5).round(3))

best_cfg = {task: res.iloc[0] for task, res in results_net.items()}           # folosita la sarcinile 10-11
```

### Observații
```markdown
- **Configurația câștigătoare:** clasificare […]; regresie […]. De obicei nu e aceeași: clasificarea (o clasă rară, puține pozitive) preferă rețele mici sau cu Dropout, regresia (o țintă continuă, mai multă informație pe rând) poate folosi o rețea puțin mai mare.
- **Diferențele dintre configurații** (primele 5) sunt de ordinul […], comparabile cu abaterea standard dintre fold-uri ([…]): alegerea e în mare parte zgomot (sarcina 10 verifică asta).
- **Timp:** […] s pentru 24 de antrenări, față de ≈ 20 s pentru 30 de combinații × 5 fold-uri ale `RandomizedSearchCV` pe HistGB în L05: rețeaua e de ≈ […] ori mai scumpă per configurație.
```

---

## Sarcina 10 — Variația între seed-uri (celula 59) ❌

### Cod corect
```python
# TODO: configuratia aleasa, 5 seed-uri, pe setul de validare din sarcina 3
rows = []
for task, (A_f, t_f, A_v, t_v) in data.items():
    cfg = best_cfg[task]                                    # din sarcina 9: units e deja un tuplu, ex. (32, 16)
    for seed in [0, 1, 2, 3, 4]:
        keras.backend.clear_session()                       # sterge reteaua de la seed-ul anterior
        keras.utils.set_random_seed(seed)
        net = make_net(A_f.shape[1], task, units=cfg["units"], dropout=float(cfg["dropout"]), lr=float(cfg["lr"]),
                       loss="mae" if task == "regression" else None)
        net.fit(A_f, t_f, validation_data=(A_v, t_v), epochs=300, batch_size=32, verbose=0, callbacks=[early_stop()],
                class_weight=balanced_weights(t_f) if task == "classification" else None)
        p = net.predict(A_v, verbose=0).ravel()
        score = average_precision_score(t_v, p) if task == "classification" else mean_absolute_error(yr_val, to_money(p))
        rows.append({"task": task, "seed": seed, "score": score})
seeds = pd.DataFrame(rows)
display(seeds.groupby("task")["score"].agg(["mean", "std", "min", "max"]).round(3))
# comparatie: cat de departe sunt primele configuratii din sarcina 9?
for task, res in results_net.items():
    main = "PR_AUC" if task == "classification" else "MAE"
    print(f"{task}: spread of the top-5 configurations = {res[main].head(5).max() - res[main].head(5).min():.3f}")
```

### Observații
```markdown
- **Variația între seed-uri:** abaterea standard pe 5 seed-uri e […] (clasificare) și […] (regresie); diferența dintre cel mai bun și cel mai slab seed e […].
- **Comparație:** diferența dintre primele 5 configurații din sarcina 9 e […]. Dacă variația între seed-uri e la fel de mare sau mai mare, **clasamentul configurațiilor e zgomot**: o altă inițializare a ponderilor le-ar ordona altfel.
- **Ce înseamnă pentru alegere:** nu aleg configurația după un singur scor. Aleg o configurație simplă din grupul de top și raportez media pe mai multe seed-uri (sau fac media predicțiilor mai multor rețele, ca un mic ansamblu).
```

---

## Sarcina 11 — Rețelele finale, salvate și reîncărcate (celula 63) ❌

### Cod corect
```python
# SARCINA: retelele finale pe tot train-ul, salvate (.keras + preprocesarea cu joblib), reincarcate si verificate
final_nets, final_preps = {}, {}
for task, X, y, num_cols in [("clf", Xc_train, yc_train, num_cols_c), ("reg", Xr_train, yr_train, num_cols_r)]:
    full_task = "classification" if task == "clf" else "regression"
    cfg = best_cfg[full_task]
    units = cfg["units"]
    # 15% din train pentru EarlyStopping; preprocesarea invatata pe restul
    X_f, X_es, y_f, y_es = train_test_split(X, np.asarray(y), test_size=0.15, random_state=SEED,
                                            stratify=y if task == "clf" else None)
    prep = make_prep(num_cols).fit(X_f)
    A_f, A_es = prep.transform(X_f).astype("float32"), prep.transform(X_es).astype("float32")
    if task == "reg":
        ys = StandardScaler().fit(y_f.reshape(-1, 1))
        t_f, t_es = (ys.transform(d.reshape(-1, 1)).ravel() for d in (y_f, y_es))
        final_preps[task] = {"prep": prep, "y_scaler": ys}
    else:
        t_f, t_es = y_f, y_es
        final_preps[task] = {"prep": prep, "threshold": threshold_net}
    keras.utils.set_random_seed(SEED)
    net = make_net(A_f.shape[1], full_task, units=units, dropout=cfg["dropout"], lr=cfg["lr"],
                   loss="mae" if task == "reg" else None)
    net.fit(A_f, t_f, validation_data=(A_es, t_es), epochs=300, batch_size=32, verbose=0, callbacks=[early_stop()],
            class_weight=balanced_weights(t_f) if task == "clf" else None)
    net.save(f"{MODELS_DIR}/L06_{MY_NAME}_net_{task}.keras")
    joblib.dump(final_preps[task], f"{MODELS_DIR}/L06_{MY_NAME}_prep_{task}.joblib")
    final_nets[task] = net

# reincarcare si verificare pe cateva randuri din train: aceleasi predictii
for task, X in [("clf", Xc_train), ("reg", Xr_train)]:
    net_loaded = keras.saving.load_model(f"{MODELS_DIR}/L06_{MY_NAME}_net_{task}.keras")
    saved = joblib.load(f"{MODELS_DIR}/L06_{MY_NAME}_prep_{task}.joblib")
    A = saved["prep"].transform(X.head(5)).astype("float32")
    before = final_nets[task].predict(A, verbose=0).ravel()
    after = net_loaded.predict(A, verbose=0).ravel()
    if task == "reg":
        before, after = (saved["y_scaler"].inverse_transform(v.reshape(-1, 1)).ravel() for v in (before, after))
    print(task, "| same predictions after reload:", np.allclose(before, after), "|", np.round(after, 3))
```

Salvez împreună cu preprocesarea și pragul (clasificare) sau scalerul țintei (regresie): fără ele, rețeaua reîncărcată nu poate fi folosită pe date noi.

---

## Sarcina 12 — Găsiți greșelile (Keras, celula 66) ❌

```markdown
**Greșelile găsite** (bloc pentru o clasificare cu 3 clase):
1. `prep.fit(X)` pe **tot** setul (inclusiv testul): leakage. Corect: `prep.fit(X_train)` (ideal, doar pe partea de antrenare fără validare).
2. `layers.Dense(3, activation="sigmoid")`: pentru 3 clase care se exclud reciproc, ieșirea trebuie să fie **`softmax`** (probabilități care se adună la 1). Sigmoid tratează fiecare clasă separat (multi-label).
3. `loss="mse"` pentru clasificare: pierderea corectă e **`sparse_categorical_crossentropy`** (etichete 0, 1, 2) sau `categorical_crossentropy` (etichete one-hot). MSE pe probabilități învață greu și nu e o pierdere de clasificare.
4. `epochs=500` fără `EarlyStopping`: rețeaua face overfitting. Corect: `callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)]`.
5. `validation_data=(prep.transform(X_test), y_test)`: **testul** e folosit ca validare (pentru monitorizare / oprire): testul nu mai e independent. Corect: un set de validare scos din train.
6. `accuracy_score(y_test, pred)` cu `pred` = **probabilități** (o matrice n × 3): dă eroare. Corect: `pred.argmax(axis=1)`.
7. `accuracy_score` nu e importat (`from sklearn.metrics import accuracy_score`): **eroare** (`NameError`).
8. Lipsesc conversia la `float32` și verificarea că `y_train` are etichete 0..K−1 (dacă sunt text, Keras dă eroare; se codifică întâi cu `LabelEncoder`).
9. (Mai puțin grav) acuratețea simplă pe clase posibil dezechilibrate; mai bine F1 macro / acuratețe echilibrată.

**Varianta corectă:**
    from sklearn.metrics import accuracy_score, f1_score
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED)
    prep.fit(X_fit)
    model = keras.Sequential([keras.Input(shape=(prep.transform(X_fit[:1]).shape[1],)),
                              layers.Dense(64, activation="relu"),
                              layers.Dense(3, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(prep.transform(X_fit).astype("float32"), y_fit, epochs=500,
              validation_data=(prep.transform(X_val).astype("float32"), y_val),
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
    pred = model.predict(prep.transform(X_test).astype("float32")).argmax(axis=1)   # o singura data, la final
    print(accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro"))
```

---

## Contribuția (celula 68) ❌

**Ipoteză:** un **ansamblu de 5 rețele** (aceeași configurație, seed-uri diferite, media probabilităților / predicțiilor) e mai bun și mai stabil decât o singură rețea, pentru că rețelele cu inițializări diferite greșesc pe exemple diferite (L05, TODO 3). În sarcina 10 am văzut că o singură rețea depinde mult de seed.

### Cod corect
```python
# CONTRIBUTIA: o retea vs. media a 5 retele (seed-uri diferite), pe setul de validare din sarcina 3
def val_score(task, p, t_v):
    """Metrica principala pe validare: PR-AUC (clasificare) / MAE in unitati monetare (regresie)."""
    return average_precision_score(t_v, p) if task == "classification" else mean_absolute_error(yr_val, to_money(p))

rows = []
for task, (A_f, t_f, A_v, t_v) in data.items():
    cfg = best_cfg[task]                                    # din sarcina 9: units e deja un tuplu, ex. (32, 16)
    preds = []
    for seed in range(5):
        keras.backend.clear_session()                       # sterge reteaua de la seed-ul anterior
        keras.utils.set_random_seed(seed)
        net = make_net(A_f.shape[1], task, units=cfg["units"], dropout=float(cfg["dropout"]), lr=float(cfg["lr"]),
                       loss="mae" if task == "regression" else None)
        net.fit(A_f, t_f, validation_data=(A_v, t_v), epochs=300, batch_size=32, verbose=0, callbacks=[early_stop()],
                class_weight=balanced_weights(t_f) if task == "classification" else None)
        preds.append(net.predict(A_v, verbose=0).ravel())
    single = [val_score(task, p, t_v) for p in preds]
    rows.append({"task": task, "single net (mean of 5)": np.mean(single), "single net (std)": np.std(single),
                 "ensemble of 5 nets": val_score(task, np.mean(preds, axis=0), t_v)})   # media predictiilor celor 5 retele
display(pd.DataFrame(rows).set_index("task").round(3))
```

> Atenție: setul de validare a fost folosit și pentru `EarlyStopping`, deci ambele scoruri sunt ușor optimiste în aceeași măsură. Comparația între ele rămâne corectă. Pentru o estimare curată, pune media celor 5 rețele în `net_cv`.

### Observații
```markdown
- **Ipoteza:** media a 5 rețele bate o singură rețea.
- **Experimentul:** aceeași configurație (cea mai bună din sarcina 9), 5 seed-uri, același set de validare; compar media scorurilor rețelelor individuale cu scorul mediei predicțiilor lor.
- **Rezultatul:** clasificare: o rețea […] ± […], ansamblul […]; regresie: o rețea […] ± […], ansamblul […].
- **Concluzia:** […] (de așteptat: ansamblul e cel puțin la nivelul celei mai bune rețele individuale și elimină dependența de seed, cu un cost de 5 ori mai mare în timp. Dacă diferența e sub variația dintre seed-uri, câștigul principal e **stabilitatea**, nu scorul).
```

---

## Celulele AI

Toate sunt goale. Completează-le cinstit dacă ai folosit un asistent.
