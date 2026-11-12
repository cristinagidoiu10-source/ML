# Teorie — Laboratorul 6 (bonus): rețele neuronale cu Keras vs. scikit-learn

> Adaptat după `lab6/TEORIE_LAB6.md` din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB) (Telco), refăcut pe **Customer Personality Analysis**. Codul e cel din `L06_raport.md`. Rețelele depind de seed și de hardware, deci multe cifre le obții tu la rulare; cele fixe (dimensiuni, parametri, ponderi) sunt calculate aici.

În Lab 6 construim **rețele neuronale** cu Keras pentru aceleași două probleme (`Response`, `MntTotal`) și le comparăm cu modelele din Lab 5 (regresie logistică PR-AUC 0,601; HistGB MAE 185,5). Întrebarea de fond: **merită o rețea pe un tabel mic?**

**Datele pentru rețea:** din cei 1646 de clienți de train, 20% (≈ 330) devin **set de validare** pentru antrenarea rețelei → ≈ **1316** rânduri de antrenare. Testul rămâne închis.

**Cuprins**
1. Ce e o rețea neuronală
2. Funcțiile de activare
3. Numărul de parametri
4. Pregătirea datelor pentru Keras
5. Pierderea (loss)
6. Antrenarea: gradient, backpropagation, Adam, epoci, batch
7. `compile`, `fit`, `predict`, istoricul
8. Overfitting și regularizare
9. Dezechilibrul la rețele
10. CV și căutarea hiperparametrilor pentru o rețea
11. Reproductibilitatea
12. Salvarea și reîncărcarea
13. Keras vs. scikit-learn (MLP)
14. Contribuția: media mai multor rețele
15. Când merită o rețea?
16. **Codul din Lab 6, explicat**
17. Rezumat + întrebări

---

## 1. Ce e o rețea neuronală

### Neuronul (artificial)
Primește mai multe numere, face o **sumă ponderată** + un **bias**, apoi aplică o **funcție de activare**:
`ieșire = activare(w₁·x₁ + w₂·x₂ + … + b)`.
Fără activare, un neuron e exact o regresie liniară; cu sigmoidă la final, e o regresie logistică.

### Stratul (layer)
Mai mulți neuroni care primesc aceleași intrări. **Strat de intrare** (atributele), **straturi ascunse**, **strat de ieșire** (răspunsul).

### Strat `Dense` (complet conectat)
Fiecare neuron e legat de **toate** ieșirile stratului anterior.

### MLP (perceptron multistrat)
O rețea doar din straturi `Dense`. E tipul potrivit pentru date tabelare.

### Arhitectura
Câte straturi și câți neuroni. **La noi:** prima rețea **16 → 8 → 1** (mică, pentru că avem doar ~1316 rânduri).

### Keras
Biblioteca (peste TensorFlow / PyTorch / JAX) cu care definim rețeaua strat cu strat.

---

## 2. Funcțiile de activare

| Activare | Formula | Unde |
|---|---|---|
| **ReLU** | max(0, z) | straturile ascunse (simplă, rapidă, nu „saturează” pentru z > 0) |
| **sigmoid** | 1 / (1 + e⁻ᶻ) ∈ (0, 1) | ieșirea la **clasificare binară** → P(`Response` = 1) |
| **liniară** (niciuna) | z | ieșirea la **regresie** → orice număr (`MntTotal` scalat) |
| softmax | probabilități care însumează 1 | clasificare cu mai multe clase (nu la noi) |

Fără activări neliniare, oricâte straturi ar fi echivalente cu unul singur (o compunere de funcții liniare e liniară).

---

## 3. Numărul de parametri

Un strat `Dense` cu **n** intrări și **u** neuroni are **n·u ponderi + u bias-uri**.

**La noi:**
- clasificare: **30** de coloane după preprocesare (23 numerice + `Education` + 6 one-hot `Marital_Status`);
  30·16 + 16 = 496; 16·8 + 8 = 136; 8·1 + 1 = 9 → **641** de parametri;
- regresie: **20** de coloane (13 numerice + 1 + 6) → 20·16+16 + 136 + 9 = **481**.

641 de parametri la 1316 rânduri ≈ **1 parametru la 2 rânduri** → rețeaua poate memora ușor un set atât de mic → are nevoie de oprire timpurie / regularizare. Rețeaua „mare” (256-256-128) are ~**107.000** de parametri — de 80 de ori mai mulți decât rândurile.

---

## 4. Pregătirea datelor pentru Keras

Keras nu acceptă `NaN`, text sau DataFrame-uri cu tipuri amestecate. Are nevoie de o **matrice de numere `float32`**:
1. aceeași preprocesare ca în L03–L05 (mediană, scalare, ordinal, one-hot), **învățată doar pe partea de antrenare** (`X_fit`);
2. `.astype("float32")`;
3. la **regresie**, scalăm și **ținta** (media 0, std 1): `MntTotal` are std ~600; MSE pe sume de bani ar porni de la sute de mii și learning rate-ul implicit n-ar funcționa;
4. predicțiile de regresie se transformă înapoi în bani (`to_money`) înainte de a calcula MAE.

---

## 5. Pierderea (loss)

Ce minimizează rețeaua la antrenare.
| Problemă | Pierdere | Ce înseamnă |
|---|---|---|
| clasificare binară | **binary cross-entropy** | −[y·log p + (1−y)·log(1−p)]: penalizează mult o probabilitate mare dată clasei greșite |
| regresie | **MSE** | media erorilor la pătrat; sensibilă la erorile mari |
| regresie | **MAE** | media erorilor absolute; robustă |
| regresie | **Huber** | MSE pentru erori mici, MAE pentru erori mari |

**La noi (experimentul din laborator):** pe o țintă cu **5% erori simulate** (+5 std), MSE se strică cel mai mult (eroarea la pătrat dă greutate uriașă valorilor greșite), MAE și Huber rămân aproape la fel. **Alegem MAE** (sau Huber): metrica noastră e MAE, iar `MntTotal` are clienți foarte mari care la MSE ar trage modelul după ei.

---

## 6. Antrenarea

### Gradientul și coborârea pe gradient
**Gradientul** = direcția în care pierderea crește cel mai repede. Mutăm ponderile puțin în direcția **opusă** → pierderea scade. Repetăm.

### Backpropagation
Algoritmul care calculează eficient gradientul pentru **toate** ponderile, pornind de la ieșire înapoi spre intrare (regula lanțului din derivate).

### Optimizatorul Adam
Coborâre pe gradient „inteligentă”: fiecare pondere are propriul pas, adaptat după istoricul gradienților. Implicitul bun pentru aproape orice.

### Rata de învățare (learning rate)
Mărimea pasului. Prea mare → oscilează / diverge; prea mică → învață foarte încet. **La noi:** 0,001 (implicit Adam) și 0,003 în căutare.

### Epocă
O trecere completă prin toate rândurile de antrenare. **La noi:** max 200–300, oprite mai devreme de `EarlyStopping`.

### Batch și `batch_size`
Ponderile se actualizează după fiecare **grup** de rânduri (batch), nu după tot setul. **La noi:** `batch_size=32` → ~41 de actualizări pe epocă (1316 / 32).

---

## 7. `compile`, `fit`, `predict` și istoricul

- `compile(optimizer, loss, metrics)` — ce optimizator, ce pierdere, ce metrici afișează;
- `fit(X, y, validation_data=(X_val, y_val), epochs, batch_size)` — antrenarea; întoarce un **`history`**;
- `predict(X)` — sigmoida dă **probabilități** (pragul îl alegem noi), regresia dă valori scalate.

### Cum citești curbele din `history`
`history.history["loss"]` și `["val_loss"]` = pierderea pe fiecare epocă.
- ambele scad → încă învață;
- train scade, validarea **stagnează și apoi crește** → **overfitting** de acolo înainte;
- minimul lui `val_loss` = epoca ideală de oprire.

**La noi:** cu rețeaua 16-8 și 200 de epoci, pierderea pe validare scade în primele zeci de epoci, apoi crește: rețeaua începe să memoreze cei ~1316 clienți.

---

## 8. Overfitting la rețele și regularizarea

### Oprirea timpurie (`EarlyStopping`)
Oprește antrenarea când `val_loss` nu mai scade de `patience=20` epoci; `restore_best_weights=True` revine la epoca cu cea mai mică pierdere. Aceeași idee ca la HistGB (L05: 67 / 104 arbori) — acolo „pasul” e un arbore, aici o epocă.

### Dropout
La fiecare pas de antrenare, „oprește” aleator o fracțiune (de ex. 30%) din neuroni. Rețeaua nu se mai poate baza pe un singur neuron → învață reguli mai robuste. La `predict`, Dropout e dezactivat.

### Regularizarea L2
Adaugă la pierdere λ·Σw² → penalizează ponderile mari → model mai „neted” (aceeași idee ca `C` la regresia logistică, invers: λ mare = regularizare puternică).

**Experimentul din laborator:** rețeaua mare (256-256-128, ~107k parametri) fără regularizare rulează 300 de epoci și pierderea pe validare crește mult → overfitting clar. Cu **EarlyStopping**, **+ Dropout 0,3**, **+ L2 1e-3** overfitting-ul e controlat; la seturi atât de mici, ES + Dropout sau ES + L2 bat de obicei ES singur.

---

## 9. Dezechilibrul la rețele

`fit(..., class_weight={0: w0, 1: w1})` — la fel ca `class_weight="balanced"` în scikit-learn.
**La noi:** `compute_class_weight("balanced")` → **≈ 0,59 pentru clasa 0** și **≈ 3,3 pentru clasa 1** (n / (2·n_clasă)).
Efect (prag 0,5, pe validare): recall-ul crește mult, precizia scade, iar **PR-AUC rămâne aproape același** — ponderile schimbă pragul efectiv, nu ordinea clienților. Același efect ca în L03 (recall 0,41 → 0,78).
`MLPClassifier` din scikit-learn **nu are** `class_weight` → una dintre diferențele față de Keras.

---

## 10. Validarea încrucișată și căutarea hiperparametrilor pentru o rețea

### CV pentru o rețea
Keras nu se integrează direct cu `cross_validate`, așa că scriem bucla de mână. În **fiecare fold**:
1. preprocesarea (și scalarea țintei) se învață doar pe partea de antrenare a fold-ului;
2. 15% din partea de antrenare → pentru `EarlyStopping`;
3. o rețea **nouă** (altfel ar continua de unde a rămas în fold-ul anterior);
4. scorul pe fold-ul de validare.
Aceleași fold-uri ca în L05 → putem face testul t corectat față de regresia logistică / HistGB.

### Căutarea
**Grila:** arhitectura `(16,)` sau `(32, 16)` × Dropout 0 / 0,3 × lr 0,001 / 0,003 = **8 configurații** × **3 fold-uri** (nu 5: e scump) = 24 de antrenări pe problemă. Pe CPU, câteva minute — față de ~20 s pentru 150 de antrenări HistGB în L05.
**Ce te aștepți să vezi:** diferențele dintre primele 5 configurații sunt comparabile cu std-ul dintre fold-uri → alegerea e în mare parte zgomot.

---

## 11. Reproductibilitatea

Rețeaua pornește din **ponderi aleatoare** și amestecă batch-urile aleator → același cod dă scoruri diferite la fiecare rulare.
- `keras.utils.set_random_seed(SEED)` fixează Python, NumPy și backend-ul;
- `keras.backend.clear_session()` șterge rețelele vechi din memorie între fold-uri / seed-uri;
- pe GPU unele operații rămân nedeterministe.

**Experimentul din laborator:** aceeași configurație, **5 seed-uri** → dacă std-ul între seed-uri e cât diferența dintre configurații, **clasamentul configurațiilor e zgomot**. Raportăm media pe mai multe seed-uri, nu un singur scor.

---

## 12. Salvarea și reîncărcarea modelului

- rețeaua: `net.save("model.keras")` → `keras.models.load_model(...)`;
- preprocesarea (un obiect scikit-learn): **separat**, cu `joblib.dump`;
- la regresie și **scalerul țintei** (altfel nu poți transforma predicțiile în bani).

Verificare: după reîncărcare, predicțiile pe câteva rânduri trebuie să fie **identice** cu cele dinainte.

---

## 13. Keras vs. scikit-learn (MLP)

| | Keras | `MLPClassifier` / `MLPRegressor` |
|---|---|---|
| definirea | strat cu strat, orice arhitectură | `hidden_layer_sizes=(32, 16)` |
| Dropout, BatchNorm | da | nu (doar L2 prin `alpha`) |
| `class_weight` | da | **nu** |
| pierderi | orice (MAE, Huber, proprii) | fixe (log-loss / MSE) |
| integrare cu pipeline / CV | manuală | directă (`make_pipe`, `cross_validate`) |
| GPU | da | nu |

Cu aceeași arhitectură, scorurile sunt de obicei apropiate; diferențele la recall vin în principal din `class_weight`.

---

## 14. Contribuția: media a 5 rețele

**Ipoteza:** media predicțiilor a 5 rețele (seed-uri diferite) bate o singură rețea.
**Așteptarea:** ansamblul e cel puțin la nivelul celei mai bune rețele individuale și elimină dependența de seed, cu un cost de 5× în timp. Dacă diferența e sub variația dintre seed-uri, câștigul principal e **stabilitatea**, nu scorul. (Aceeași idee ca bagging-ul din L05.)

---

## 15. Când merită o rețea?

**Pe setul nostru (~1650 de rânduri, ~25 de coloane): nu.** În cel mai bun caz egalează regresia logistică / HistGB (diferență nesemnificativă la testul t), e de zeci de ori mai lentă, depinde de seed și e greu de explicat marketingului.
Rețelele câștigă la **date mari și nestructurate** (imagini, text, sunet), unde atributele trebuie învățate. Pe tabele mici, **modelele liniare și boosting-ul pe arbori** rămân alegerea standard.

---

## 16. Codul din Lab 6, explicat

### 16.1 Datele pentru Keras

```python
Xc_fit, Xc_val, yc_fit, yc_val = train_test_split(Xc_train, yc_train, test_size=0.2, stratify=yc_train, random_state=SEED)
prep_clf = make_prep(num_cols_c).fit(Xc_fit)
Ac_fit = prep_clf.transform(Xc_fit).astype("float32")
Ac_val = prep_clf.transform(Xc_val).astype("float32")
tc_fit = yc_fit.to_numpy().astype("float32")
```
- Un set de **validare** din train (testul rămâne închis) — pentru `EarlyStopping` și alegeri.
- `make_prep` = doar partea de preprocesare (`ColumnTransformer`) din L04–L05, fără model; `fit` doar pe `Xc_fit`.
- `A` = matricea de atribute deja preprocesată; `t` = ținta ca vector `float32`.

```python
y_scaler = StandardScaler().fit(yr_fit.to_numpy().reshape(-1, 1))
tr_fit = y_scaler.transform(yr_fit.to_numpy().reshape(-1, 1)).ravel().astype("float32")

def to_money(t_scaled):
    return y_scaler.inverse_transform(np.asarray(t_scaled).reshape(-1, 1)).ravel()
```
- `StandardScaler` vrea o **matrice** (rânduri × coloane) → `reshape(-1, 1)` transformă vectorul într-o coloană; `ravel()` îl aplatizează la loc.
- `to_money` face drumul invers: predicția scalată → unități monetare.

### 16.2 Prima rețea

```python
keras.utils.set_random_seed(SEED)
net_clf = keras.Sequential([
    keras.Input(shape=(Ac_fit.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
], name="net_clf")
net_clf.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
```
- `Sequential` = straturi unul după altul. `Input(shape=(30,))` = câte coloane intră.
- Ieșire `sigmoid` → o probabilitate; pierdere `binary_crossentropy`; metrica afișată = PR-AUC (`AUC(curve="PR")`), aceeași ca în L04–L05.
- La regresie: ultimul strat `Dense(1)` fără activare, `loss="mse"`, metrică `MeanAbsoluteError`.

```python
hist_clf = net_clf.fit(Ac_fit, tc_fit, validation_data=(Ac_val, tc_val), epochs=200, batch_size=32, verbose=0)
axes[row, col].plot(hist.history[key], label="train")
axes[row, col].plot(hist.history[f"val_{key}"], label="validation")
val_loss.argmin() + 1
```
- `verbose=0` → fără afișare la fiecare epocă.
- `hist.history` = dicționar cu o listă de valori pe epocă pentru `loss`, `val_loss`, metrici → desenăm curbele.
- `argmin() + 1` = epoca cu pierderea minimă pe validare (+1 pentru că indexarea începe de la 0).

```python
def count_params(n_inputs, units):
    sizes = [n_inputs, *units]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
count_params(30, [16, 8, 1])          # 641 = net_clf.count_params()
```
- `[n_inputs, *units]` = `[30, 16, 8, 1]`; `zip(sizes[:-1], sizes[1:])` dă perechile (30,16), (16,8), (8,1) = (intrări, neuroni) pentru fiecare strat.
- Fiecare strat: `a*b` ponderi + `b` bias-uri. Verificăm calculul de mână cu `count_params()` din Keras.

### 16.3 O funcție care construiește rețele

```python
def make_net(n_inputs, task, units=(16, 8), dropout=0.0, l2=0.0, lr=1e-3, loss=None):
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
```
- Rețeaua se construiește **în buclă** după tuplul `units` → `(256, 256, 128)` sau `(32, 16)` cu același cod.
- `kernel_regularizer=regularizers.l2(...)` adaugă L2 pe ponderile stratului; `Dropout` vine după fiecare strat ascuns.
- `loss or "mse"` = dacă nu dai o pierdere, folosește implicitul.
- `early_stop()` e o **funcție** care întoarce un callback nou de fiecare dată (un callback ține stare internă, nu trebuie refolosit între antrenări).

```python
variants = {
    "big, no regularization":    dict(units=BIG, stop=False),
    "big + EarlyStopping":       dict(units=BIG, stop=True),
    "big + Dropout 0.3 + ES":    dict(units=BIG, dropout=0.3, stop=True),
    "big + L2 1e-3 + ES":        dict(units=BIG, l2=1e-3, stop=True),
}
cfg = dict(cfg); stop = cfg.pop("stop")
net = make_net(A_f.shape[1], task, **cfg)
net.fit(..., epochs=300, callbacks=[early_stop()] if stop else [])
```
- Fiecare variantă e un dicționar de argumente; `cfg.pop("stop")` scoate cheia care nu e argument pentru `make_net`; `**cfg` le dă pe restul ca argumente cu nume.
- `dict(cfg)` face o **copie** → `pop` nu strică dicționarul original (folosit și la regresie).

### 16.4 `class_weight` și pierderile robuste

```python
def balanced_weights(y):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.asarray(y).astype(int))
    return {0: w[0], 1: w[1]}
net.fit(..., class_weight=cw, callbacks=[early_stop()])
```
Keras vrea `class_weight` ca dicționar `{clasă: pondere}`; scikit-learn ni-l calculează.

```python
corrupt = rng.random(len(tr_fit)) < 0.05
tr_fit_noisy[corrupt] = tr_fit_noisy[corrupt] + rng.normal(5, 1, corrupt.sum())
for loss_name, loss in [("MSE", "mse"), ("MAE", "mae"), ("Huber", keras.losses.Huber(delta=1.0))]:
    for data_name, target in [("clean", tr_fit), ("5% errors", tr_fit_noisy)]:
        ...
pd.DataFrame(rows).pivot(index="loss", columns="training target", values="validation MAE")
```
- O mască cu ~5% True; pe acele rânduri adăugăm ~5 std la țintă (erori grosolane) — **doar în antrenare**; validarea rămâne curată.
- `pivot` transformă lista lungă de rezultate într-un tabel 3 × 2 (pierdere × date).

### 16.5 Raportul și comparația cu MLP

```python
ok = [t for t in thresholds if recall_score(tc_val, p_val >= t) >= 0.75]
threshold_net = max(ok, key=lambda t: precision_score(tc_val, p_val >= t))
print(classification_report(tc_val, (p_val >= threshold_net).astype(int), digits=3))

mlp_c = make_pipe(MLPClassifier(hidden_layer_sizes=UNITS, activation="relu", alpha=1e-3, early_stopping=True,
                                max_iter=500, random_state=SEED), num_cols_c).fit(Xc_fit, yc_fit)
```
- Același criteriu de prag ca în L04–L05. Atenție: aici pragul e ales pe validare și raportat tot pe validare → ușor optimist.
- `MLPClassifier` cu aceeași arhitectură intră direct în `make_pipe` (e un estimator scikit-learn); `alpha` = L2.

### 16.6 Bucla de CV pentru rețea

```python
def net_cv(X, y, task, cv, num_cols, units=UNITS, dropout=0.3, ..., seed=SEED):
    for idx_fit, idx_val in cv.split(X, y):
        X_f, X_v, y_f, y_v = X.iloc[idx_fit], X.iloc[idx_val], y[idx_fit], y[idx_val]
        X_f, X_es, y_f, y_es = train_test_split(X_f, y_f, test_size=0.15, ...)
        prep = make_prep(num_cols).fit(X_f)
        A_f, A_es, A_v = (prep.transform(d).astype("float32") for d in (X_f, X_es, X_v))
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        net = make_net(A_f.shape[1], task, units=units, dropout=dropout, ...)
        net.fit(A_f, t_f, validation_data=(A_es, t_es), ..., callbacks=[early_stop()])
        p = net.predict(A_v, verbose=0).ravel()
        rows.append({"PR_AUC": average_precision_score(y_v, p), ...})
```
- `cv.split(X, y)` dă indicii fiecărui fold; `iloc` selectează rândurile după poziție.
- În fold: încă o împărțire 85/15 pentru `EarlyStopping` → fold-ul de validare rămâne **complet nevăzut**.
- Preprocesarea și scalerul țintei se învață **în fold** → fără leakage.
- Generatorul `(... for d in (X_f, X_es, X_v))` transformă cele trei seturi într-o linie.
- `clear_session()` + `set_random_seed` + `make_net` → rețea nouă, reproductibilă, în fiecare fold.
- Întoarce un rând de tabel + scorurile pe fold-uri → `corrected_ttest` față de modelele din L05 pe **aceleași** fold-uri.

### 16.7 Căutarea, seed-urile, salvarea

```python
grid_net = {"units": [(16,), (32, 16)], "dropout": [0.0, 0.3], "lr": [1e-3, 3e-3]}
for units, dropout, lr in product(*grid_net.values()):
    row, _ = net_cv(X, y, task, cv, num_cols, units=units, dropout=dropout, lr=lr, epochs=150)
```
`itertools.product` generează toate cele 2 × 2 × 2 = 8 combinații — un „GridSearch” scris de mână.

```python
for seed in [0, 1, 2, 3, 4]:
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    net = make_net(...)
seeds.groupby("task")["score"].agg(["mean", "std", "min", "max"])
```
Aceeași configurație, 5 inițializări diferite → cât variază scorul doar din cauza norocului.

```python
net.save(f"{MODELS_DIR}/L06_{MY_NAME}_net_clf.keras")
joblib.dump(prep, f"{MODELS_DIR}/L06_{MY_NAME}_prep_clf.joblib")
net2 = keras.models.load_model(...); prep2 = joblib.load(...)
np.allclose(net.predict(A), net2.predict(prep2.transform(X).astype("float32")))
```
Rețeaua și preprocesarea se salvează separat; după reîncărcare, `np.allclose` verifică că predicțiile sunt (aproape) identice.

---

## 17. Rezumat

| Termen | Pe scurt | La noi |
|---|---|---|
| neuron | sumă ponderată + activare | |
| Dense / MLP | complet conectat / doar Dense | 16 → 8 → 1 |
| ReLU / sigmoid / liniară | ascunse / ieșire clasificare / ieșire regresie | |
| parametri | n·u + u pe strat | 641 / 481 la ~1316 rânduri |
| float32, fără NaN | cerința Keras | preprocesarea din L03–L05 |
| țintă scalată | regresie | `MntTotal` std ~600 → 1 |
| loss | ce se minimizează | cross-entropy / MAE |
| Adam, lr, epocă, batch | cum se antrenează | 1e-3, ≤ 300, 32 |
| EarlyStopping / Dropout / L2 | contra overfitting | patience 20 |
| class_weight | clasa rară | {0: 0,59, 1: 3,3} |
| seed | reproductibilitate | 5 seed-uri → zgomot |
| verdict | merită? | nu, pe un tabel mic |

### Întrebări de verificare

1. Câți parametri are un strat `Dense(16)` cu 30 de intrări?
2. De ce ieșirea de la clasificare are sigmoid, iar cea de la regresie nu are activare?
3. De ce scalăm `MntTotal` înainte de antrenare?
4. Ce arată o curbă `val_loss` care scade apoi crește?
5. De ce `class_weight` nu schimbă mult PR-AUC?
6. De ce construim o rețea nouă în fiecare fold?
7. Ai alege rețeaua pentru campanie?

**Răspunsuri:**
1. 30 · 16 + 16 = 496.
2. Sigmoid dă o probabilitate în (0, 1); la regresie ținta poate fi orice număr.
3. Pierderea pornește de la valori uriașe și pasul optimizatorului nu se potrivește; cu media 0 / std 1, setările implicite funcționează.
4. Overfitting de la minim încolo → `EarlyStopping` revine la acea epocă.
5. Ponderile mută pragul efectiv (mai mulți clienți peste 0,5), dar nu schimbă ordinea scorurilor, pe care o măsoară PR-AUC.
6. Altfel rețeaua ar continua din ponderile fold-ului anterior, care a văzut datele de validare ale fold-ului curent → leakage.
7. Nu: în cel mai bun caz egalează regresia logistică, dar e mai lentă, instabilă între seed-uri și greu de explicat.
