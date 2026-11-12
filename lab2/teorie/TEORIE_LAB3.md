# Teorie — Laboratorul 3: preprocesarea, pipeline-ul, primele modele și validarea

> Adaptat după `lab3/TEORIE_LAB3.md` din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB) (Telco), refăcut pe **Customer Personality Analysis**. Cifrele vin din rularea codului corectat din `L03_raport.md` pe `clean.csv` (2058 de clienți). Pe Colab pot diferi la a treia zecimală.

În Lab 2 am **cunoscut și curățat** datele. În Lab 3 construim corect **drumul de la date la model**: împărțim datele, pregătim coloanele într-un **pipeline**, tratăm **dezechilibrul**, comparăm modele prin **validare încrucișată**, căutăm **hiperparametrii** și facem **o singură** evaluare pe test.

Ținta principală în acest laborator: **`Response`** (clasificare). Structura e aceeași ca în repo: fiecare termen are **Ce înseamnă / La noi / În cod**, iar la final (secțiunea 17) e **codul explicat**.

**Cuprins**
1. Antrenare, test, validare
2. Modele, parametri, hiperparametri
3. Overfitting, underfitting, bias, varianță
4. `fit` / `transform` / `predict`
5. Imputarea valorilor lipsă
6. Scalarea
7. Codificarea categoriilor
8. `ColumnTransformer` și `Pipeline`
9. Data leakage prin preprocesare
10. Dezechilibrul claselor și reechilibrarea
11. Metricile de clasificare
12. Primele modele
13. Validarea încrucișată
14. Curbele de diagnostic
15. Căutarea hiperparametrilor
16. Testul final și pragul de decizie
17. **Codul din Lab 3, explicat**
18. Rezumat + întrebări

---

## 1. Antrenare, test, validare

### Setul de antrenare (train)
**Ce înseamnă:** rândurile din care modelul **învață**.
**La noi:** 80% din 2058 → **1646** clienți (din care ~269 cu `Response = 1`).

### Setul de test
**Ce înseamnă:** rândurile **ascunse** până la final, folosite **o singură dată**, ca să estimăm cum se va comporta modelul pe clienți noi. Dacă ne uităm la test și apoi schimbăm modelul, testul nu mai e cinstit.
**La noi:** **412** clienți, din care doar **63** cu `Response = 1`. De aceea scorul pe test e „zgomotos”: fiecare client din clasa 1 ghicit sau ratat mută recall-ul cu 1/63 ≈ 1,6 puncte.

### Setul de validare
**Ce înseamnă:** o parte din **train** pe care comparăm modele și alegem setări. Testul rămâne închis.
**La noi:** nu păstrăm un set de validare separat (2058 de rânduri sunt puține) — folosim **validarea încrucișată** (secțiunea 13).

### Split-ul
**În cod:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)`.
- `stratify=y`: aceeași proporție de 1 (~15%) în train și în test. Fără ea, testul ar putea avea, din întâmplare, 10% sau 20% clasa 1;
- `random_state=SEED`: aceeași împărțire la fiecare rulare (rezultate reproductibile);
- **aleator, nu cronologic:** `Dt_Customer` e data înscrierii, nu momentul predicției; datele nu sunt o serie de timp.

---

## 2. Modele, parametri, hiperparametri

### Model
O „formulă” cu forma stabilită dinainte (o dreaptă, un arbore…) și cu valori învățate din date.

### Parametri
**Ce înseamnă:** numerele pe care modelul le **învață singur** în `fit`.
**La noi:** la regresia logistică, câte o **pondere** pentru fiecare atribut (de ex. cât de mult crește `AcceptedCmp5` șansa de răspuns).

### Hiperparametri
**Ce înseamnă:** setări pe care le alegem **noi**, înainte de antrenare. Modelul nu le poate învăța singur.
**La noi:** `C` la regresia logistică, `max_depth` / `min_samples_leaf` la arbori, `k` la kNN, `class_weight`.

### Antrenare (fit)
Procesul prin care modelul își găsește parametrii pe datele de antrenare.

---

## 3. Overfitting, underfitting, bias, varianță

### Overfitting (supra-învățare)
**Ce înseamnă:** modelul **memorează** train-ul (inclusiv zgomotul) și greșește pe date noi. Semnal: scor foarte mare pe train, mult mai mic pe validare.
**La noi:** arborele fără limită: train **0,998** / validare **0,610** (acuratețe echilibrată). Gradient boosting: 0,998 / 0,718.

### Underfitting (sub-învățare)
**Ce înseamnă:** modelul e **prea simplu** și nu prinde nici regulile din train. Semnal: ambele scoruri mici.
**La noi:** regresia logistică cu `C = 0,001` (regularizare foarte puternică): train 0,669 / validare 0,665.

### Bias și varianță
- **Bias** = eroarea din cauza unui model prea simplu (underfitting);
- **Varianță** = cât de mult se schimbă modelul dacă schimbi puțin datele (overfitting).
**La noi:** regresia logistică are train 0,829 / validare 0,802 → diferență mică (varianță mică), dar ambele ~0,80 → e limitată de **bias** (e liniară). Arborele adânc are varianță mare.

### Generalizare
Cât de bine merge modelul pe **date noi**. Tot ce facem în Lab 3 e ca să **estimăm cinstit** generalizarea.

---

## 4. `fit` / `transform` / `predict`

| Metodă | Ce face | Pe ce date |
|---|---|---|
| `fit(X, y)` | **învață** (mediana, media/std la scalare, categoriile, ponderile) | **doar train** |
| `transform(X)` | **aplică** ce a învățat | train, validare, test |
| `fit_transform(X)` | ambele, într-un pas | doar train |
| `predict(X)` | dă clasa prezisă (0/1) | orice |
| `predict_proba(X)` | dă probabilitatea fiecărei clase | orice |

**Regula:** `fit` **niciodată** pe test. Mediana lui `Income` se calculează din train și se aplică și pe test.

---

## 5. Imputarea (completarea valorilor lipsă)

### Imputare
**Ce înseamnă:** înlocuim `NaN` cu o valoare estimată.
**În cod:** `SimpleImputer(strategy="median")` (numerice), `strategy="most_frequent"` (categoriale).
**La noi:** în train lipsesc **22** de valori la `Income` și **2** la `Year_Birth`. Mediana e preferată: `Income` e asimetric (și fusese 666666), iar mediana nu e trasă de extreme. Practic, la atât de puține lipsuri, media și mediana dau scoruri identice.

### `add_indicator`
**Ce înseamnă:** pe lângă completare, adaugă o coloană 0/1 „lipsea valoarea?”. Util dacă **faptul că lipsește** spune ceva.
**La noi:** clienții fără `Income` din train au răspuns în **4,5%** din cazuri față de **15,3%** la ceilalți — dar sunt doar 22 de rânduri (un singur client a răspuns), deci diferența poate fi întâmplătoare. **Lăsăm căutarea (GridSearch) să decidă**: alege `add_indicator=True`, cu câștig mic.

---

## 6. Scalarea

### Ce e și de ce e nevoie
Atributele au **unități** foarte diferite: `Income` ~50.000, `Kidhome` 0–2, `Recency` 0–99. Modelele bazate pe **distanțe** (kNN, SVM) sau pe **ponderi regularizate** (regresia logistică) ar fi dominate de `Income`. Scalarea le aduce la aceeași „mărime”.

| Scaler | Formula | Când |
|---|---|---|
| `StandardScaler` | (x − medie) / std | implicit |
| `MinMaxScaler` | (x − min) / (max − min) → [0, 1] | sensibil la outlieri |
| `RobustScaler` | (x − mediană) / IQR | **avem outlieri** |
| `"passthrough"` | nimic | **arbori** (nu au nevoie: pun praguri, nu calculează distanțe) |

**La noi:** am ales **`RobustScaler`** ca implicit (outlieri în `Income`, `Mnt*`, `NumDealsPurchases`); căutarea a ales până la urmă `StandardScaler`, dar diferența e neglijabilă.

---

## 7. Codificarea categoriilor

Modelele lucrează doar cu **numere**. Coloanele text trebuie transformate.

### One-hot
**Ce înseamnă:** o coloană 0/1 pentru fiecare categorie.
**La noi:** `Marital_Status` (nominală) → `Married`, `Together`, `Single`, `Divorced`, `Widow`, `Other`.
- `handle_unknown="infrequent_if_exist"` / `"ignore"`: o categorie nouă în test nu dă eroare;
- `min_frequency=10`: categoriile cu sub 10 apariții în train (`Other`, 4 rânduri) sunt adunate într-o singură coloană „infrequent”.

### Codificarea ordinală
**Ce înseamnă:** fiecare categorie primește un număr, **în ordinea** dată de noi.
**La noi:** `Education` → `Basic = 0 < Graduation = 1 < Master = 2 < PhD = 3`. (`2n Cycle` a fost unit cu `Master` la curățare; `OrdinalEncoder` nu poate da aceeași valoare la două categorii.)
**Atenție:** fără `categories=...`, `OrdinalEncoder` le ordonează **alfabetic** (`2n Cycle < Basic < Graduation…`), fără sens.

### Coloanele 0/1
`AcceptedCmp1–5`, `Complain` sunt deja 0/1 → rămân numerice. `Kidhome` / `Teenhome` sunt numărători → numerice.

---

## 8. `ColumnTransformer` și `Pipeline`

### `ColumnTransformer`
**Ce înseamnă:** aplică **transformări diferite pe grupuri diferite de coloane** și lipește rezultatul.
**La noi:** 3 ramuri: `num` (imputare + scalare), `ord` (`Education`), `nom` (`Marital_Status`).

### `Pipeline`
**Ce înseamnă:** un lanț de pași (preprocesare → model) care se comportă ca **un singur model**: `pipe.fit(X_train, y_train)`, `pipe.predict(X_test)`.
**De ce:** (1) nu poți uita un pas pe test; (2) în validarea încrucișată preprocesarea se învață **separat pe fiecare fold** → fără leakage; (3) modelul se salvează complet (`joblib.dump`).

### `imblearn.Pipeline`
Ca `Pipeline`, dar permite un pas care **schimbă numărul de rânduri** (oversampling, SMOTE). Pasul de reeșantionare se aplică **doar la `fit`**, nu și la `predict`.

---

## 9. Data leakage prin preprocesare

**Ce înseamnă:** preprocesarea „vede” datele de test/validare. Exemplu: calculezi mediana lui `Income` sau media pentru scalare pe **tot** setul, apoi împarți. Atunci testul a influențat antrenarea.
**Soluția:** **toată** preprocesarea stă în pipeline și se învață doar pe partea de antrenare (a split-ului sau a fiecărui fold).
**Greșeală clasică:** `SMOTE` aplicat **înainte** de `cross_validate` → rânduri sintetice create din clienții de validare ajung în antrenare.

---

## 10. Dezechilibrul claselor și reechilibrarea

**La noi:** 85% / 15%, raport ~5,7. Fără reechilibrare, regresia logistică găsește doar **41%** dintre clienții care acceptă.

| Strategie | Ce face |
|---|---|
| `class_weight="balanced"` | greșelile pe clasa rară „costă” mai mult (~5,7× la noi) în funcția de pierdere |
| Random oversampling | copiază rânduri din clasa rară (risc: memorare) |
| SMOTE | creează rânduri **sintetice** între vecini din clasa rară (risc: clienți nerealiști) |
| Undersampling | aruncă rânduri din clasa mare (pierdem date) |

**La noi (CV, regresie logistică):**
| strategie | recall | precizie | F1 | bal. acc. |
|---|---|---|---|---|
| none | 0,412 | 0,673 | 0,501 | 0,687 |
| **class_weight** | **0,784** | 0,431 | **0,555** | **0,798** |
| oversampling | 0,780 | 0,433 | 0,555 | 0,798 |
| SMOTE | 0,748 | 0,441 | 0,553 | 0,788 |

**Alegerea:** `class_weight="balanced"` — la egalitate cu oversampling (diferențe ≤ 0,01, sub std ≈ 0,04), dar cea mai simplă: nu creează și nu copiază rânduri.
**Prețul:** precizia scade de la 0,67 la 0,43. Pentru campanie e un compromis bun: un contact inutil costă 3, un client ratat înseamnă 11 pierduți.

---

## 11. Metricile de clasificare

### Matricea de confuzie
|  | prezis 0 | prezis 1 |
|---|---|---|
| **real 0** | TN (corect respins) | **FP** (contactat degeaba) |
| **real 1** | **FN** (client interesat ratat) | TP (găsit) |

### Metricile
| Metrică | Formula | Întrebarea | La noi |
|---|---|---|---|
| Acuratețe | (TP+TN)/total | câte am ghicit? | înșală: 85% cu „mereu 0” |
| Precizie | TP/(TP+FP) | din cei contactați, câți acceptă? | costul contactelor |
| Recall | TP/(TP+FN) | din cei interesați, pe câți îi găsim? | venitul ratat |
| F1 | 2·P·R/(P+R) | echilibru precizie–recall | |
| Acuratețe echilibrată | (recall₀ + recall₁)/2 | media recall-urilor pe clase | **metrica principală în L03** (baseline = 0,5) |
| ROC-AUC | aria sub curba ROC | cât de bine **ordonează** clienții după probabilitate | nu depinde de prag |

---

## 12. Primele modele

### Baseline (`DummyClassifier`)
Prezice mereu clasa majoritară. Orice model trebuie să-l bată. La noi: acuratețe 0,85, dar bal. acc. 0,5, F1 0.

### Regresia logistică
Calculează o **sumă ponderată** a atributelor și o trece prin funcția sigmoidă → o probabilitate între 0 și 1. Hiperparametrul **`C`** = inversul regularizării: C mic = ponderi mici, model simplu; C mare = model liber.
**La noi:** cel mai bun model (bal. acc. CV **0,798 ± 0,039**, ROC-AUC 0,884).

### kNN (k vecini)
Clasifică un client după cei **k clienți cei mai asemănători** din train. Are nevoie de scalare. Nu are `class_weight` → la noi prezice aproape mereu 0 (bal. acc. 0,529, F1 0,125).

### Arborele de decizie
O serie de întrebări da/nu („`AcceptedCmp5 ≤ 0,5`?”, „`Recency ≤ 25`?”). Ușor de interpretat, dar fără limite memorează train-ul (0,998 / 0,610). Hiperparametri: `max_depth`, `min_samples_leaf`.

### Random forest
Mulți arbori pe eșantioane diferite, apoi vot (detalii în Lab 5). La noi: 0,741 ± 0,035; ROC-AUC 0,870 — ordonează aproape la fel de bine ca regresia logistică, dar pragul 0,5 nu i se potrivește.

---

## 13. Validarea încrucișată (CV)

### Problema cu o singură validare
Cu 1646 de rânduri și ~269 de pozitive, scorul pe o singură împărțire depinde mult de noroc.

### k-fold
Împărțim train-ul în **k = 5** bucăți (fold-uri). De 5 ori: antrenăm pe 4, validăm pe a 5-a. Raportăm **media ± abaterea standard** a celor 5 scoruri.
- **`StratifiedKFold`**: fiecare fold are ~15% clasa 1;
- **`shuffle=True, random_state=SEED`**: aceleași fold-uri pentru **toate** modelele → diferențele vin din modele, nu din împărțire.

### Cum citești rezultatul
Diferența dintre două modele e credibilă doar dacă e **mai mare decât abaterea standard**. La noi: logistică 0,798 vs random forest 0,741 → diferență 0,057 ≈ 1,5 × std → probabil reală.

### În cod
`cross_validate(pipe, X_train, y_train, cv=cv5, scoring=[...], return_train_score=True)`.

---

## 14. Curbele de diagnostic

### Curba de validare
Scorul pe train și pe validare în funcție de **un hiperparametru**.
**La noi (`C` al regresiei logistice):** C = 0,001 → 0,669 / 0,665 (underfitting); crește până la C = 1 (validare **0,798**), apoi rămâne plată (0,802 la C = 10, 0,803 la C = 100). Aleg **C = 1**: dintre valorile cu același scor, modelul cel mai simplu.

### Curba de învățare
Scorul în funcție de **câte date** de antrenare folosim.
**La noi:** cu 131 de exemple, train 0,933 / validare 0,695 (memorează). La 1316 exemple: 0,829 / 0,796. Curbele s-au apropiat și sunt amândouă ~0,80 → **mai multe date ar ajuta puțin**; ar ajuta mai mult **atribute noi** (modelul e limitat de bias).

### Scara logaritmică
`C`, `gamma`, `alpha` se caută pe **puteri ale lui 10** (0,001, 0,01, 0,1, 1, 10, 100): efectul lor e multiplicativ. `ax.set_xscale("log")` la grafic.

---

## 15. Căutarea hiperparametrilor

### Grid search (`GridSearchCV`)
Încearcă **toate combinațiile** dintr-o grilă, fiecare cu CV, și o păstrează pe cea mai bună. Numele parametrilor au prefixul pasului din pipeline: `model__C`, `prep__num__scale`, `prep__num__impute__add_indicator`.
**La noi:**
- regresie logistică: 5 × 2 × 2 × 2 = **40** de combinații × 5 fold-uri = 200 de antrenări → cel mai bun: `C = 1`, `class_weight="balanced"`, `StandardScaler`, `add_indicator=True`: **0,808 ± 0,033** (câștig +0,010 față de setările inițiale, sub std → setările erau deja bune);
- random forest: 3 × 4 × 2 = **24** de combinații → `min_samples_leaf=10`, `max_features="sqrt"`: 0,741 → **0,772** (overfitting redus: train 0,984 → 0,923).

### `best_score_` e puțin optimist
Am ales **maximul** din 40 de scoruri → cel ales a avut și puțin noroc cu fold-urile.

### Nested CV
O CV **în interiorul** altei CV: bucla interioară caută hiperparametrii, bucla exterioară evaluează procedura „caută + antrenează” pe fold-uri nevăzute.
**La noi:** regresia logistică **0,798 ± 0,038** (sub `best_score_` 0,808, exact cum spune teoria); random forest 0,753 ± 0,019. **Alegem regresia logistică.**

---

## 16. Testul final și pragul de decizie

### Testul final
O singură evaluare, cu modelul ales (`grid_lr.best_estimator_`, reantrenat pe tot train-ul).
**La noi:** test **bal. acc. 0,845**, F1 0,617, ROC-AUC 0,917. Testul iese puțin peste nested CV (0,798), dar diferența (~1 std) e zgomot: doar 63 de pozitive.
**Față de L02** (0,753): **+0,092**, venit aproape în întregime din **`class_weight`** (recall 0,52 → 0,86; precizie 0,85 → 0,48).

### Pragul de decizie
`predict` folosește pragul **0,5** pe probabilitate. Putem alege alt prag, dar **doar din train** (predicții *out-of-fold*), nu pe test.
**La noi:** pragul care maximizează F1 pe OOF = **0,60** (peste 0,5, pentru că `class_weight` a împins deja probabilitățile în sus). Pe OOF câștigul e mic (0,566 → 0,570); pe test F1 urcă 0,617 → 0,684, dar e în mare parte noroc.

---

## 17. Codul din Lab 3, explicat

### 17.1 Citirea și split-ul

```python
CLEAN_CSV = f"{MY_DIR}/clean.csv"
assert os.path.exists(CLEAN_CSV), f"Nu exista {CLEAN_CSV}: rulati intai capitolul 5 din laboratorul 2"
df = pd.read_csv(CLEAN_CSV)
X = df.drop(columns=[TARGET, TARGET_R])
y = df[TARGET]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
```
- Pornim de la **setul curățat** în L02 (o singură sursă). `assert` dă un mesaj clar dacă fișierul lipsește.
- `X` fără ambele ținte (`Response` și `MntTotal`). `MntTotal` = suma `Mnt*`, redundantă.
- Listele de coloane se calculează **automat** după tip, nu de mână.
- De aici `X_test` / `y_test` **nu se mai ating** până la capitolul 5.

### 17.2 Pipeline-ul de preprocesare

```python
ORD_COLS = ["Education"]
NOM_COLS = [c for c in cat_cols if c not in ORD_COLS]              # Marital_Status
EDU_ORDER = [["Basic", "Graduation", "Master", "PhD"]]

def make_pipeline(model, scaler=None, num_strategy="median", add_indicator=False):
    num_steps = Pipeline([("impute", SimpleImputer(strategy=num_strategy, add_indicator=add_indicator)),
                          ("scale", scaler if scaler is not None else RobustScaler())])
    ord_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=EDU_ORDER,
                                                     handle_unknown="use_encoded_value", unknown_value=-1))])
    nom_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist"))])
    prep = ColumnTransformer([("num", num_steps, num_cols),
                              ("ord", ord_steps, ORD_COLS),
                              ("nom", nom_steps, NOM_COLS)])
    return Pipeline([("prep", prep), ("model", model)])
```
- O **funcție** care primește un model și întoarce pipeline-ul complet → aceeași preprocesare pentru toate modelele.
- `scaler=None` → `RobustScaler` implicit; `scaler="passthrough"` → fără scalare (pentru arbori).
- `EDU_ORDER` e o **listă de liste** (câte o listă de ordine pentru fiecare coloană ordinală).
- `handle_unknown="use_encoded_value", unknown_value=-1`: o categorie nouă de `Education` devine −1, fără eroare.
- Numele pașilor (`prep`, `num`, `impute`, `scale`, `model`) contează: din ele se fac numele din grilă (`prep__num__scale`).

```python
no_income = X_train["Income"].isna()
y_train[no_income].mean(), y_train[~no_income].mean()
```
Verificare pentru `add_indicator`: rata de răspuns la clienții **cu** și **fără** venit, doar pe train. `~` = negația măștii.

```python
pipe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
pipe.named_steps["prep"].get_feature_names_out()
```
Verifică că pipeline-ul rulează și arată numele coloanelor după preprocesare (de ex. `num__Income`, `ord__Education`, `nom__Marital_Status_Single`).

### 17.3 Compararea strategiilor de dezechilibru

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strategies = {
    "none": make_pipeline(LogisticRegression(max_iter=2000)),
    "class_weight": make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "oversampling": ImbPipeline([("prep", prep), ("resample", RandomOverSampler(random_state=SEED)),
                                 ("model", LogisticRegression(max_iter=2000))]),
    "SMOTE": ImbPipeline([("prep", prep), ("resample", SMOTE(random_state=SEED)),
                          ("model", LogisticRegression(max_iter=2000))]),
}
for name, pipe in strategies.items():
    s = cross_validate(pipe, X_train, y_train, cv=cv5,
                       scoring=["recall", "precision", "f1", "balanced_accuracy"])
```
- **Același** `cv5` pentru toate → comparație corectă.
- Reeșantionarea stă **în** `ImbPipeline`, după `prep` → se face doar pe partea de antrenare a fiecărui fold. Fold-ul de validare rămâne cu proporția reală.
- SMOTE vine **după** preprocesare: are nevoie de numere fără `NaN`, ca să calculeze distanțe.

### 17.4 Compararea modelelor prin CV

```python
models = {
    "baseline": make_pipeline(DummyClassifier(strategy="most_frequent")),
    "logistic regression": make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "kNN": make_pipeline(KNeighborsClassifier()),
    "decision tree": make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=SEED), scaler="passthrough"),
    "random forest": make_pipeline(RandomForestClassifier(class_weight="balanced", min_samples_leaf=3, random_state=SEED),
                                   scaler="passthrough"),
}
rows = {}
for name, pipe in models.items():
    s = cross_validate(pipe, X_train, y_train, cv=cv5, scoring=["balanced_accuracy", "f1", "roc_auc"],
                       return_train_score=True)
    rows[name] = {"val bal. acc. (mean)": s["test_balanced_accuracy"].mean(),
                  "val bal. acc. (std)": s["test_balanced_accuracy"].std(),
                  "train bal. acc.": s["train_balanced_accuracy"].mean(), ...}
cv_results = pd.DataFrame(rows).T.sort_values("val bal. acc. (mean)", ascending=False)
```
- `cross_validate` întoarce un dicționar: `test_<metrica>` = scorurile pe cele 5 fold-uri de **validare** (numele „test” e al lui scikit-learn, nu e testul nostru!), `train_<metrica>` = pe partea de antrenare.
- `return_train_score=True` → putem vedea overfitting-ul (train mult peste validare).
- Tabelul sortat după media pe validare → cel mai bun model sus.

### 17.5 Curbele

```python
from sklearn.model_selection import validation_curve, learning_curve
C_range = np.logspace(-3, 2, 6)                                 # 0.001 ... 100
train_vc, val_vc = validation_curve(make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
                                    X_train, y_train, param_name="model__C", param_range=C_range,
                                    cv=cv5, scoring="balanced_accuracy")
ax.plot(C_range, train_vc.mean(axis=1), label="train")
ax.plot(C_range, val_vc.mean(axis=1), label="validation")
ax.fill_between(C_range, val_vc.mean(1) - val_vc.std(1), val_vc.mean(1) + val_vc.std(1), alpha=0.2)
ax.set_xscale("log")

sizes, train_lc, val_lc = learning_curve(best_pipe, X_train, y_train, cv=cv5,
                                         train_sizes=np.linspace(0.1, 1.0, 5), scoring="balanced_accuracy")
```
- `np.logspace(-3, 2, 6)` = 6 valori între 10⁻³ și 10² pe scară logaritmică.
- `validation_curve` întoarce matrici (valoare × fold); `.mean(axis=1)` = media pe fold-uri.
- `fill_between` desenează banda ± std → vezi cât de sigure sunt diferențele.
- `learning_curve` antrenează pe 10%, 32%, …, 100% din train și raportează scorurile.

### 17.6 GridSearch și nested CV

```python
param_grid_lr = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"],
    "prep__num__scale": [StandardScaler(), RobustScaler()],
    "prep__num__impute__add_indicator": [False, True],
}
grid_lr = GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000)), param_grid_lr, cv=cv5,
                       scoring="balanced_accuracy", n_jobs=-1, return_train_score=True)
grid_lr.fit(X_train, y_train)
grid_lr.best_params_, grid_lr.best_score_
```
- Grila poate conține și **pași de preprocesare** (`prep__num__scale` primește chiar obiecte scaler) → alegem preprocesarea tot prin CV.
- `n_jobs=-1` → folosește toate nucleele procesorului.
- După `fit`, `best_estimator_` e pipeline-ul cu cei mai buni parametri, **reantrenat pe tot X_train**.

```python
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
inner = GridSearchCV(grid_lr.estimator, grid_lr.param_grid, cv=inner_cv, scoring=SCORING, n_jobs=-1)
nested_scores = cross_val_score(inner, X_train, y_train, cv=cv5, scoring=SCORING)
```
- `inner` e o căutare completă; `cross_val_score` o rulează pe fiecare din cele 5 fold-uri exterioare → 5 scoruri ale **procedurii** „caută + antrenează”.
- Folosim **aceeași** grilă (`grid_lr.param_grid`) → estimăm exact procedura folosită.

### 17.7 Testul final

```python
best_model = grid_lr.best_estimator_
y_pred_te = best_model.predict(X_test)
y_proba_te = best_model.predict_proba(X_test)[:, 1]
balanced_accuracy_score(y_test, y_pred_te), f1_score(y_test, y_pred_te), roc_auc_score(y_test, y_proba_te)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_te, cmap="Blues")
joblib.dump(best_model, f"{MODELS_DIR}/L03_{MY_NAME}_best.joblib")
```
- O **singură** evaluare pe test, cu aceleași metrici ca în CV.
- `joblib.dump` salvează **tot pipeline-ul** (preprocesare + model) → la reîncărcare, `model.predict(date_noi_brute)` funcționează direct.

### 17.8 Pragul ales din OOF

```python
proba_oof = cross_val_predict(best_model, X_train, y_train, cv=cv5, method="predict_proba")[:, 1]
thresholds = np.arange(0.05, 0.96, 0.05)
f1_by_t = [f1_score(y_train, (proba_oof >= t).astype(int)) for t in thresholds]
best_t = thresholds[int(np.argmax(f1_by_t))]
y_pred_t = (y_proba_te >= best_t).astype(int)
```
- `cross_val_predict` dă, pentru **fiecare** rând din train, probabilitatea prezisă de modelul care **nu l-a văzut** (out-of-fold) → putem alege pragul fără test.
- `(proba >= t).astype(int)` transformă probabilitățile în clase cu pragul `t`.
- `np.argmax` = poziția valorii maxime → pragul cu cel mai mare F1. Abia apoi îl aplicăm o dată pe test.

---

## 18. Rezumat

| Termen | Pe scurt | La noi |
|---|---|---|
| train / test | învățare / evaluare finală | 1646 / 412 (63 pozitive) |
| stratify | păstrează proporția claselor | ~15% clasa 1 peste tot |
| hiperparametru | setare aleasă de noi | `C`, `max_depth`, `class_weight` |
| overfitting | train ≫ validare | arbore: 0,998 / 0,610 |
| underfitting | ambele mici | logistică C = 0,001: 0,67 / 0,67 |
| imputare | completare `NaN` | mediana pentru `Income` |
| scalare | aceeași „mărime” | `RobustScaler` / `StandardScaler` |
| one-hot / ordinal | categorie → coloane 0/1 / număr ordonat | `Marital_Status` / `Education` |
| pipeline | preprocesare + model = un obiect | fără leakage în CV |
| class_weight | clasa rară costă mai mult | recall 0,41 → 0,78 |
| bal. acc. | media recall-urilor | metrica principală |
| k-fold CV | 5 împărțiri, medie ± std | logistică 0,798 ± 0,039 |
| GridSearch | toate combinațiile | 40 de combinații → 0,808 |
| nested CV | estimarea cinstită a căutării | 0,798 ± 0,038 |
| test | o singură dată | 0,845 bal. acc., ROC-AUC 0,917 |

### Întrebări de verificare

1. De ce mediana lui `Income` se învață doar pe train?
2. De ce `Education` e codificată ordinal, iar `Marital_Status` one-hot?
3. Random forest are train 0,984 și validare 0,741. Ce înseamnă?
4. De ce SMOTE se pune în `ImbPipeline` și nu se aplică înainte de CV?
5. `best_score_` = 0,808, nested CV = 0,798. De ce diferă?
6. De ce pragul de decizie nu se alege pe test?
7. De ce arborii primesc `scaler="passthrough"`?

**Răspunsuri:**
1. Altfel informația din test (sau din fold-ul de validare) intră în antrenare → leakage, scor optimist.
2. `Education` are ordine (Basic < … < PhD), `Marital_Status` nu: un număr ar inventa o ordine falsă.
3. Overfitting: memorează train-ul. Soluția: frunze mai mari (`min_samples_leaf=10`) → 0,772.
4. Rândurile sintetice ar fi create și din clienții de validare → validarea ar „vedea” antrenarea.
5. `best_score_` e maximul din 40 de scoruri (include noroc); nested CV evaluează procedura pe fold-uri nevăzute.
6. Pragul ar fi ajustat chiar pe datele pe care raportăm → scor optimist; testul nu mai e cinstit.
7. Arborii pun praguri pe câte o coloană (`Income ≤ 60.000`); scalarea nu schimbă ordinea valorilor, deci nici arborele.
