# Teorie — Laboratorul 5: ansambluri, selecția atributelor, testul final și interpretarea

> Adaptat după `lab5/TEORIE_LAB5.md` din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB) (Telco), refăcut pe **Customer Personality Analysis**. Cifrele vin din codul din `L05_raport.md` (CV pe train; testul o singură dată, la final).

În Lab 5 combinăm mai multe modele într-unul singur (**ansambluri**), vedem dacă merită să păstrăm doar o parte din atribute (**selecție**), deschidem **testul** o singură dată și **interpretăm** modelul final (ce atribute contează, cât câștigă campania).

Aceleași două probleme ca în Lab 4: `Response` (metrica principală **PR-AUC**) și `MntTotal` (metrica principală **MAE**).

**Scoruri de referință (Lab 4):** regresie logistică PR-AUC 0,601, SVM 0,606, baseline 0,152; regresie liniară MAE 233, SVR 199, baseline 527.

**Cuprins**
1. Ideea ansamblurilor
2. Bootstrap și bagging
3. Random forest și extra-trees
4. Scorul out-of-bag (OOB)
5. Boosting
6. Rata de învățare, numărul de arbori, oprirea timpurie
7. Voting și stacking
8. Dezechilibrul la ansambluri
9. Transformarea logaritmică a țintei
10. `RandomizedSearchCV`
11. Selecția atributelor
12. Testul final și intervalele de încredere
13. Curba de câștig și lift-ul
14. Importanța atributelor
15. Metrica de business (contribuția)
16. **Codul din Lab 5, explicat**
17. Rezumat + întrebări

---

## 1. Ideea ansamblurilor

### Ansamblu
**Ce înseamnă:** mai multe modele care „votează” sau se corectează reciproc; rezultatul combinat e de obicei mai bun decât fiecare.

### „Înțelepciunea mulțimii” și votul majoritar
Dacă avem M modele **independente**, fiecare corect cu probabilitatea p > 0,5, votul majoritar e corect mai des decât fiecare.
**Calculul din laborator:**
1. 5 modele independente, p = 0,7 → votul e corect cu **0,837** (> 0,7);
2. 3 din 5 sunt **copii** ale aceluiași model → **0,700**: cele 3 copii formează singure majoritatea, ansamblul nu mai aduce nimic;
3. p = 0,45 (mai slabe decât o monedă) → probabilitatea **scade** cu M (0,45 → 0,16 la M = 101): votul amplifică și greșelile.

### Diversitate
Un ansamblu ajută doar dacă membrii sunt (1) **mai buni decât întâmplarea** și (2) **diferiți** (greșesc pe clienți diferiți). De aceea random forest alege atribute la întâmplare, iar la voting/stacking combinăm familii diferite (liniar, SVM, arbori).

---

## 2. Bootstrap și bagging

### Eșantion bootstrap
**Ce înseamnă:** tragem **n** rânduri din n, **cu înlocuire**. Unele rânduri apar de mai multe ori, ~37% nu apar deloc (rândurile „out-of-bag”).

### Bagging (bootstrap aggregating)
**Ce înseamnă:** antrenăm câte un model (de obicei un arbore adânc) pe fiecare eșantion bootstrap și facem media / votul. Media reduce **varianța** (overfitting-ul), nu bias-ul.

**La noi:**
| Model | Clasificare PR-AUC | Regresie MAE |
|---|---|---|
| un singur arbore | 0,324 (train 0,897) | 233,5 |
| bagging (300 de arbori) | 0,527 | 192 |
| random forest | 0,566 | 192 |
| extra-trees | **0,584** | **187,3** |

Media arborilor aproape **dublează** PR-AUC și scade MAE cu ~45 (≈ 4 abateri standard). Pe train arborele singur și ansamblurile au aceeași eroare la regresie (126); diferența e pe validare → s-a redus **varianța**.

---

## 3. Random forest și extra-trees

### Random forest
Bagging + la **fiecare împărțire** arborele poate alege doar dintr-un subset aleator de atribute (`max_features`). Arborii devin mai diferiți → media lor e mai bună.

### Extra-trees (extremely randomized trees)
Ca random forest, dar și **pragul** de împărțire se alege la întâmplare (nu cel optim). Și mai multă diversitate, mai rapid. La noi e cel mai bun dintre ansamblurile de tip bagging.

---

## 4. Scorul out-of-bag (OOB)

**Ce înseamnă:** fiecare rând e prezis doar de arborii care **nu l-au văzut** (~1/3 dintre ei) → o validare „gratuită”, fără CV (`oob_score=True`).
**La noi (`max_features`):**
- clasificare: cel mai bun OOB la **0,3** (PR-AUC 0,599) → mai puține atribute pe împărțire = arbori mai diferiți;
- regresie: la **0,5** (MAE OOB 186,2) → `Income` trebuie să fie disponibil des (e de departe cel mai important).
OOB ≈ CV la regresie (diferențe 3–4); la clasificare OOB e cu 0,03–0,05 peste CV, dar **ordinea** valorilor se păstrează → bun pentru o alegere rapidă.

---

## 5. Boosting

### Ideea
Arborii se construiesc **unul după altul**; fiecare nou arbore încearcă să **corecteze greșelile** celor de dinainte. Folosește arbori mici (slabi); reduce mai ales **bias-ul**.

### AdaBoost
Mărește ponderea rândurilor greșit clasificate, ca următorul arbore să se concentreze pe ele.

### Gradient Boosting
Fiecare arbore nou învață **reziduurile** (gradientul pierderii) modelului de până atunci.

### HistGradientBoosting
Gradient boosting care grupează valorile atributelor în ~255 de „găleți” (histograme) → **mult mai rapid**; acceptă `NaN` nativ și are oprire timpurie.

**La noi:**
| | Clasificare PR-AUC | Regresie MAE | Timp / antrenare |
|---|---|---|---|
| random forest | 0,566 ± 0,040 | 192,2 | ~0,9 s |
| GradientBoosting | **0,598** ± 0,032 (dar recall 0,384 — n-are `class_weight`) | **182,7** | ~1,4 s |
| HistGB (oprire timpurie) | 0,553 | 185,5 | **~0,4 s** |

---

## 6. Rata de învățare, numărul de arbori, oprirea timpurie

### Rata de învățare (learning rate, η)
Cât din corectura fiecărui arbore adăugăm. η mic → pași mici, trebuie mai mulți arbori, dar curba e mai netedă și generalizează mai bine.
**La noi (minimul pierderii pe validare):**
| η | Clasificare | Regresie |
|---|---|---|
| 0,3 | 29 de arbori | 89 (curbă zgomotoasă) |
| 0,1 | 137 | 42 |
| 0,03 | 378 | 408 |
η × numărul de arbori ≈ constant.

### Oprirea timpurie (early stopping)
Se oprește când pierderea pe o parte de validare (din train) nu mai scade. **La noi:** HistGB a ales **67** de arbori la clasificare și **104** la regresie (din max 1000).

### Funcția de pierdere (loss)
Ce minimizează boosting-ul: `log_loss` la clasificare; la regresie `squared_error` sau **`absolute_error`** (MAE — ce am folosit, ca să optimizeze exact metrica noastră și să fie robust la clienții foarte mari).

---

## 7. Voting și stacking

### Voting
Media predicțiilor membrilor. **Soft** = media probabilităților (mai bun); **hard** = vot pe clase.

### Stacking
Un **meta-model** învață cum să combine predicțiile membrilor (predicții out-of-fold, ca să nu învețe din predicții „văzute”).

**Membrii (familii diferite):** clasificare — regresie logistică, SVM (calibrat), HistGB; regresie — regresie liniară, SVR, random forest.
**La noi:**
| | Clasificare PR-AUC | Regresie MAE |
|---|---|---|
| cel mai bun membru | 0,609 (SVM calibrat) | 192,2 (random forest) |
| voting | **0,630** ± 0,048 | 193,0 (tras în jos de regresia liniară) |
| stacking | 0,625 ± 0,044 | **186,1** |

Câștigul (+0,02) e **sub** o abatere standard. Stacking-ul e cel mai scump (5–9 s pe antrenare: fiecare membru de 6 ori).
**Ponderile meta-modelului:** clasificare — logistică 2,71 și HistGB 2,73, SVM doar 0,85 (spune același lucru ca logistica); regresie — random forest 0,61, SVR 0,37, liniară 0,02 (ignorată).

---

## 8. Dezechilibrul la ansambluri

| Strategie | Recall | F1 | PR-AUC |
|---|---|---|---|
| RF fără ponderi | **0,160** | — | 0,569 |
| RF `class_weight="balanced"` | 0,636 | 0,523 | ~0,57 |
| `BalancedRandomForest` (fiecare arbore pe un eșantion echilibrat) | **0,680** | 0,542 | 0,545 |

Fără ponderi, random forest **ordonează** bine (PR-AUC aproape același), dar pragul 0,5 nu i se potrivește → ratează 84% dintre clienții interesați. Aleg `class_weight="balanced"`: același PR-AUC, recall de 4 ori mai mare.

---

## 9. Transformarea logaritmică a țintei

**Ce înseamnă:** antrenăm pe `log(1 + y)` și transformăm predicția înapoi cu `exp(p) − 1`. Ajută când ținta e puternic asimetrică.
**La noi:** `MntTotal` are skew **0,87** (moderat). Rezultat:
| Model | Țintă originală | Țintă log |
|---|---|---|
| random forest | 192,2 | 192,3 (egal) |
| HistGB | 185,5 | 200,4 (mai rău) |
| regresie liniară | 233 | **394, R² −19** (catastrofal) |

**De ce nu ajută:** arborii sunt insensibili la transformări monotone; pe scara log se optimizează eroarea **relativă**, dar noi măsurăm MAE absolut (dominat de clienții mari); la regresia liniară `expm1` amplifică exponențial orice extrapolare. → **Nu transformăm ținta.**

---

## 10. Căutarea aleatoare (`RandomizedSearchCV`)

**Ce înseamnă:** în loc să încerce toate combinațiile (grid), încearcă **n_iter** combinații trase la întâmplare din **distribuții**. Cu același buget, explorează mai multe valori diferite pentru fiecare parametru.
- `loguniform(0.01, 0.3)` → valori uniforme pe scară logaritmică (pentru `learning_rate`, `l2_regularization`);
- `randint(8, 65)` → întregi (pentru `max_leaf_nodes`, `min_samples_leaf`).

**La noi (HistGB, 30 de încercări × 5 fold-uri):**
- clasificare: 0,553 → **0,603** (+0,050, peste std). Cheia: **`min_samples_leaf = 71`** (frunze mari → nu mai memorează). Chiar reglat, doar ajunge regresia logistică (0,601);
- regresie: 185,5 → 182,5 (−3, sub std 11,3) → setările implicite erau deja bune.

---

## 11. Selecția atributelor

**De ce:** mai puține atribute = model mai simplu, mai rapid, mai ușor de explicat, uneori mai puțin overfitting.
**Trei familii:**
- **Filtru:** notează fiecare atribut independent de model (corelație, informație mutuală) → `SelectKBest(k)`;
- **Încorporată (embedded):** modelul însuși alege (L1 pune ponderi exact 0; importanța din arbori) → `SelectFromModel`;
- **Wrapper:** încearcă subseturi și antrenează modelul pe fiecare (RFE) — scump.

**Regula:** selecția e un **pas în pipeline** → se reînvață în fiecare fold. Altfel e leakage (cu zgomot pur poți obține 0,78 în loc de 0,50).

**La noi:**
- **Filtru (informația mutuală):** clasificare — cel mai bun e **k = toate** (0,601); cu 5 atribute 0,345. Regresie — k = 20 (toate).
- **L1 (clasificare):** păstrează **18 din 30**, PR-AUC 0,607 (egal). Elimină `Income`, `Kidhome`, majoritatea `Mnt*` (păstrează `MntMeatProducts`, reprezentantul lor) — atributele sunt corelate între ele.
- **Importanța RF (regresie):** păstrează **2 din 20** (`Income` = 80% din importanță) → MAE mai rău.

**Decizie:** **fără selecție** în modelul final — nu îmbunătățește semnificativ, iar avem doar ~25 de atribute.

---

## 12. Testul final și intervalele de încredere

### Pașii
1. Alegem modelul final din tabelul CV (la scor egal → cel mai simplu);
2. alegem pragul pe **OOB din train**;
3. antrenăm pe **tot** train-ul;
4. evaluăm **o singură dată** pe test.

**Modelele finale:** clasificare → **regresia logistică** (egală cu voting-ul, p = 0,275, dar de 20× mai rapidă, probabilități directe, ponderi explicabile); regresie → **HistGB** (primul în tabel, de 20× mai rapid decât stacking-ul).

### Bootstrap pe test
Reeșantionăm rândurile testului (cu înlocuire) de 2000 de ori și recalculăm metrica de fiecare dată.

### Intervalul de încredere 95%
Percentilele 2,5 și 97,5 ale celor 2000 de valori → intervalul în care „probabil” e scorul real.

**La noi:**
| | CV | **Test** | IC 95% |
|---|---|---|---|
| clasificare, PR-AUC | 0,601 | **0,744** | 0,633 – 0,857 |
| regresie, MAE | 185,5 | **160,0** | 144,7 – 176,0 |

Clasificare pe test (prag 0,53): recall 0,857, precizie 0,535, F1 0,659, ROC-AUC 0,917. Regresie: R² 0,826.
Testul iese **mai bun** decât CV-ul — testul pare „mai ușor”; cu doar 63 de pozitive, câțiva clienți schimbă mult scorul. Intervalul larg (0,63–0,86) arată exact această incertitudine. **Nu am modificat nimic după test.**

---

## 13. Curba de câștig și lift-ul

### Curba de câștig (cumulative gains)
Ordonăm clienții după probabilitate; pe x — ce procent contactăm, pe y — ce procent din cei care acceptă am găsit.

### Lift
Lift la x% = (procentul găsit) / x%. Lift 1 = la întâmplare.
**La noi:** contactând doar **primii 20%** clienți, găsim **79%** dintre cei care acceptă → **lift@20% = 3,97**. Cu o cincime din costul campaniei, obținem patru cincimi din răspunsuri.

---

## 14. Importanța atributelor

### Importanța prin permutare
**Ce înseamnă:** amestecăm aleator valorile unei coloane (pe test) și vedem **cât scade scorul**. Mult = coloana contează. Funcționează pentru orice model.
**La noi:**
- **clasificare:** `Recency` (0,117), `AcceptedCmp5` (0,096), `MntMeatProducts` (0,091), `AcceptedCmp3`, `NumCatalogPurchases`, `Teenhome`, `Customer_Days`, `Marital_Status`. Clienții recenți, care au mai acceptat oferte și cumpără carne (produse scumpe) răspund mai des;
- **regresie:** `Income` domină (MAE crește cu **366** dacă îl amestecăm), apoi `Customer_Days` (37), `Kidhome` (22), `Teenhome` (17).

### MDI (mean decrease in impurity)
Importanța „internă” a arborilor: cât reduce fiecare atribut impuritatea, însumat pe toate împărțirile. Rapidă, dar **favorizează atributele continue cu multe valori**.
**La noi (random forest, regresie):** `Income` 0,81, `Customer_Days` 0,07… Aceeași ordine ca la permutare, dar MDI îi dă importanță și lui `Recency` (0,016), care la permutare nu contează.

---

## 15. Metrica de business (contribuția din laborator)

Setul are chiar costul și venitul campaniei: `Z_CostContact = 3`, `Z_Revenue = 11`.
**Profit** = 11 × (contactați care acceptă) − 3 × (contactați).

| Strategie | Profit / client (train OOF) | Profit / client (test) |
|---|---|---|
| contactăm pe toată lumea | **−1,33** (0,15 × 11 < 3) | pierdere |
| prag max profit (0,59) = prag max F1 | **+0,52** | **+0,74** |
| prag recall ≥ 0,75 (0,53) | +0,49 | +0,71 |

**Concluzie:** modelul transformă o campanie în pierdere într-una profitabilă. Ipoteza „pragul după profit e mai mic decât cel după F1” e respinsă: sunt identice. Cu `class_weight="balanced"` probabilitățile sunt „umflate” (antrenate ca pentru 50/50), deci 0,59 pe scara modelului corespunde unei probabilități reale mult mai mici, aproape de pragul teoretic **3/11 ≈ 0,27**.

---

## 16. Codul din Lab 5, explicat

### 16.1 Bagging, random forest, extra-trees

```python
N_TREES = 300
trees_c = {
    "one tree":    DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=SEED),
    "bagging":     BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced"),
                                     n_estimators=N_TREES, random_state=SEED, n_jobs=-1),
    "random forest": RandomForestClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced", ...),
    "extra-trees": ExtraTreesClassifier(N_TREES, min_samples_leaf=5, class_weight="balanced", ...),
}
table_bag_c, _ = compare({k: make_pipe(m, num_cols_c, scaler="passthrough") for k, m in trees_c.items()},
                         Xc_train, yc_train, "classification", cv_c)
```
- Patru variante cu **același** tip de arbore → diferențele vin doar din „media arborilor” și din aleatorism.
- `BaggingClassifier(arbore, n_estimators=300)` = 300 de copii ale arborelui, fiecare pe un eșantion bootstrap.
- **Dict comprehension** `{k: make_pipe(m, ...) for k, m in trees_c.items()}` pune fiecare model în pipeline; `scaler="passthrough"` pentru arbori.
- `compare` și `evaluate` sunt funcțiile din Lab 4.

```python
rf_c = make_pipe(RandomForestClassifier(N_TREES, max_features=mf, ..., oob_score=True, ...), ...).fit(Xc_train, yc_train)
average_precision_score(yc_train, rf_c["model"].oob_decision_function_[:, 1])
mean_absolute_error(yr_train, rf_r["model"].oob_prediction_)
```
- `oob_score=True` → după `fit`, modelul are `oob_decision_function_` (probabilitățile OOB, clasificare) sau `oob_prediction_` (regresie) pentru fiecare rând din train.
- Comparăm scorul OOB cu cel CV pentru fiecare `max_features`.

### 16.2 Boosting și curbele

```python
"HistGB (early stop)": make_pipe(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, early_stopping=True,
                                                                class_weight="balanced", random_state=SEED), ...)
"GradientBoosting": make_pipe(GradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, n_estimators=300,
                                                        max_depth=3, subsample=0.8, random_state=SEED), ...)
```
- `max_iter=1000` + `early_stopping=True` → HistGB pune un plafon și se oprește singur.
- `max_depth=3` → arbori mici (slabi), cum cere boosting-ul. `subsample=0.8` → fiecare arbore vede 80% din rânduri (puțin aleatorism = mai puțin overfitting).
- `loss="absolute_error"` → regresia optimizează direct MAE.

```python
gb_c = make_pipe(GradientBoostingClassifier(learning_rate=lr, n_estimators=1000, max_depth=3, ...), ...).fit(Xc_f, yc_f)
A_v = gb_c["prep"].transform(Xc_v)
loss_c = [log_loss(yc_v, p[:, 1]) for p in gb_c["model"].staged_predict_proba(A_v)]
best = int(np.argmin(loss))
```
- `staged_predict_proba` dă predicțiile **după 1, 2, …, 1000 de arbori** → curba pierderii pe validare în funcție de numărul de arbori, dintr-o singură antrenare.
- Validarea `Xc_v` trece întâi prin `prep` (modelul din interiorul pipeline-ului primește date deja preprocesate).
- `np.argmin(loss)` = poziția minimului → numărul optim de arbori.

```python
hgb = table["HistGB (early stop)"].fit(X, y)
hgb["model"].n_iter_
```
`n_iter_` = câți arbori a păstrat oprirea timpurie.

### 16.3 Voting și stacking

```python
members_c = {
    "logistic": make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), num_cols_c),
    "SVM (L04)": CalibratedClassifierCV(svc_l04, method="sigmoid", cv=3),
    "HistGB": boost_c["HistGB (early stop)"],
}
combos_c = {
    "voting (soft)": VotingClassifier(list(members_c.items()), voting="soft", n_jobs=-1),
    "stacking": StackingClassifier(list(members_c.items()), final_estimator=LogisticRegression(class_weight="balanced"), cv=5),
}
table_comb_c, folds_comb_c = compare(members_c | combos_c, Xc_train, yc_train, "classification", cv_c)
```
- Membrii sunt **pipeline-uri complete** → fiecare își face singur preprocesarea.
- SVM-ul e calibrat: voting-ul „soft” are nevoie de `predict_proba`.
- `list(members_c.items())` → lista de perechi `(nume, model)` cerută de scikit-learn.
- `final_estimator` = meta-modelul; `cv=5` = stacking-ul antrenează membrii pe 5 fold-uri interne ca să obțină predicții OOF pentru meta-model.
- `members_c | combos_c` = unirea a două dicționare → membrii și combinațiile în același tabel.

```python
stack_c.final_estimator_.coef_[0]
```
Ponderile meta-modelului = cât se bazează pe fiecare membru.

### 16.4 Dezechilibrul și ținta log

```python
"BalancedRandomForest": BalancedRandomForestClassifier(N_TREES, sampling_strategy="all", replacement=True,
                                                       bootstrap=False, random_state=SEED, n_jobs=-1),
```
Din `imblearn`: fiecare arbore primește un eșantion **echilibrat** (subeșantionează clasa 0).

```python
def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
```
- `func` = transformarea aplicată lui y la `fit`; `inverse_func` = cea aplicată predicțiilor.
- `log1p(y) = log(1 + y)` funcționează și pentru y = 0; `expm1` e inversa ei.

### 16.5 `RandomizedSearchCV`

```python
space = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_leaf_nodes": randint(8, 65),
    "model__min_samples_leaf": randint(10, 81),
    "model__l2_regularization": loguniform(1e-3, 10),
}
search = RandomizedSearchCV(base, space, n_iter=30, cv=cv, scoring=scoring, random_state=SEED, n_jobs=-1)
search.fit(X, y)
default = sign * cross_val_score(base, X, y, cv=cv, scoring=scoring)
```
- Spațiul conține **distribuții** din `scipy.stats`, nu liste. `randint(8, 65)` dă 8…64 (capătul drept exclus).
- `n_iter=30` = bugetul: 30 de combinații × 5 fold-uri = 150 de antrenări.
- Comparăm cu scorul **implicit** (`cross_val_score` pe modelul nereglat) → vedem cât aduce reglarea.

### 16.6 Selecția atributelor

```python
mi_c = partial(mutual_info_classif, random_state=SEED)
k_grid = {"select__k": [5, 10, 15, 20, "all"]}
filt_c = GridSearchCV(make_pipe(lr_c, num_cols_c, select=SelectKBest(mi_c)), k_grid, cv=cv_c,
                      scoring="average_precision", n_jobs=-1).fit(Xc_train, yc_train)
emb_c = make_pipe(lr_c, num_cols_c, select=SelectFromModel(
    LogisticRegression(penalty="l1", solver="liblinear", C=0.05, class_weight="balanced")))
```
- `partial` „fixează” un argument (`random_state`) al funcției → rezultate reproductibile.
- `make_pipe(..., select=...)` adaugă un pas `select` **între** `prep` și `model` (versiunea din L05 a funcției) → selecția se învață în fiecare fold.
- `k` (câte atribute păstrăm) e un hiperparametru → îl alegem prin CV.
- L1 (`penalty="l1"`) pune multe ponderi exact 0; `SelectFromModel` le păstrează pe cele nenule.

```python
names = pipe["prep"].get_feature_names_out()
kept = pipe["select"].get_support()
names[~kept]
```
`get_support()` = mască True/False pentru atributele păstrate → `names[~kept]` = cele eliminate.

### 16.7 Testul final

```python
p_oof = cross_val_predict(final_c, Xc_train, yc_train, cv=cv_c, method="predict_proba")[:, 1]
ok = [t for t in thresholds if recall_score(yc_train, p_oof >= t) >= 0.75]
THRESHOLD = max(ok, key=lambda t: precision_score(yc_train, p_oof >= t))
final_c.fit(Xc_train, yc_train)
p_te = final_c.predict_proba(Xc_test)[:, 1]
pred_c = (p_te >= THRESHOLD).astype(int)
```
- Pragul se alege **înainte** de test: dintre pragurile cu recall ≥ 0,75, cel cu precizia maximă (`max(..., key=...)`).
- `clone(...)` (mai sus) → model proaspăt, antrenat apoi pe **tot** train-ul.

```python
def bootstrap_ci(metric, y_true, y_score, n=2000):
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    values = [metric(y_true[i], y_score[i]) for i in idx]
    return np.percentile(values, [2.5, 97.5])
```
- `rng.integers(0, N, size=(2000, N))` = 2000 de rânduri de indici aleatori cu înlocuire = 2000 de „teste” bootstrap.
- Metrica pe fiecare, apoi percentilele 2,5 și 97,5 = IC 95%.

```python
order = np.argsort(-p_te)
top20 = order[: int(0.2 * len(order))]
captured = yc_test.to_numpy()[top20].sum() / yc_test.sum()
lift = captured / 0.2
```
`np.argsort(-p)` = indicii sortați **descrescător** după probabilitate; primii 20% = clienții pe care i-am contacta; `captured` = ce fracțiune din cei care acceptă sunt printre ei.

```python
imp_c = permutation_importance(final_c, Xc_test, yc_test, scoring="average_precision", n_repeats=20, random_state=SEED)
pd.Series(imp_c.importances_mean, index=Xc_test.columns).sort_values(ascending=False)
rf_mdi["model"].feature_importances_
```
- `permutation_importance` amestecă fiecare coloană de 20 de ori și raportează scăderea medie a scorului. Primește **pipeline-ul** → importanța e pe coloanele **originale** (de ex. `Marital_Status` ca un întreg, nu pe fiecare coloană one-hot).
- `feature_importances_` = MDI, pe coloanele **după** preprocesare.

### 16.8 Profitul (contribuția)

```python
REVENUE, COST = 11, 3
def profit(y_true, contacted):
    y_true, contacted = np.asarray(y_true), np.asarray(contacted)
    return REVENUE * (y_true & contacted).sum() - COST * contacted.sum()
profits = [profit(yc_train, p_oof >= t) for t in thresholds]
T_PROFIT = thresholds[int(np.argmax(profits))]
```
- `contacted` = mască True/False (cine primește oferta). `y_true & contacted` = contactați **și** acceptă → aduc 11 fiecare; fiecare contactat costă 3.
- Pragul de profit maxim e ales tot pe OOF din train, apoi verificat pe test.

---

## 17. Rezumat

| Termen | Pe scurt | La noi |
|---|---|---|
| bagging | media arborilor pe bootstrap | arbore 0,324 → 0,527 |
| random forest / extra-trees | + atribute / praguri aleatoare | 0,566 / 0,584; MAE 192 / 187 |
| OOB | validare gratuită | `max_features` 0,3 / 0,5 |
| boosting | arbori care corectează | GB 0,598; MAE 182,7 |
| learning rate + early stop | pași mici, oprire automată | 67 / 104 arbori |
| voting / stacking | combinarea familiilor | 0,630 / 0,625 (nesemnificativ) |
| log pe țintă | pentru asimetrie | nu ajută (skew 0,87) |
| RandomizedSearch | combinații aleatoare | HistGB 0,553 → 0,603 |
| selecție | mai puține atribute | nu ajută → fără |
| test + IC 95% | o singură dată + bootstrap | PR-AUC 0,744 [0,63–0,86]; MAE 160 |
| lift@20% | câștig față de întâmplare | 3,97 |
| permutare | cât scade scorul | `Recency`, `AcceptedCmp5`; `Income` |
| profit | 11 × acceptări − 3 × contacte | −1,33 → +0,52 pe client |

### Întrebări de verificare

1. De ce 3 copii ale aceluiași model nu ajută într-un vot?
2. Bagging reduce bias-ul sau varianța? Dar boosting-ul?
3. De ce learning rate mic cere mai mulți arbori?
4. Voting-ul are 0,630, regresia logistică 0,601, p = 0,275. Pe care îl alegem?
5. De ce selecția atributelor trebuie pusă în pipeline?
6. Testul dă 0,744, CV-ul 0,601. Am făcut ceva greșit?
7. Ce înseamnă lift@20% = 3,97?

**Răspunsuri:**
1. Greșesc pe aceiași clienți: formează singure majoritatea, deci votul e corect exact când modelul copiat e corect.
2. Bagging → varianța (media modelelor instabile). Boosting → mai ales bias-ul (adaugă corecturi la un model slab).
3. Fiecare arbore aduce doar o fracțiune mică din corectură → trebuie mai mulți pași ca să ajungi la același punct.
4. Regresia logistică: diferența nu e semnificativă, iar ea e mai simplă, mai rapidă și explicabilă.
5. Altfel selecția „vede” fold-urile de validare → scor optimist (leakage).
6. Nu, dacă nu am schimbat nimic după test: cu 63 de pozitive testul e zgomotos (IC 0,63–0,86); valoarea reală e probabil între cele două.
7. Contactând primii 20% clienți după scor, găsim de ~4 ori mai mulți clienți interesați decât alegând 20% la întâmplare (79% din total).
