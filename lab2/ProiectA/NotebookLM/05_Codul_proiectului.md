# Codul proiectului (Python), cu explicații

> Codul complet din notebook-ul `ProiectA_Partea_A.ipynb`, în ordinea rulării. Fiecare bloc are deasupra titlul secțiunii din care face parte. Bibliotecile principale: **pandas** (tabele), **matplotlib** (grafice), **scikit-learn** (modele ML), **imbalanced-learn** (SMOTE).


## Proiect – Partea A: Cine răspunde la campania de marketing?

**Set de date:** *Customer Personality Analysis* (Kaggle) – 2240 de clienți ai unui magazin.


## 0. Importuri și setări

`SEED = 42` peste tot → rezultatele sunt **reproductibile** (dacă rulez din nou, obțin aceleași cifre).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

SEED = 42
pd.set_option("display.max_columns", 40)
plt.rcParams["figure.figsize"] = (8, 4)
```


## 1. Formularea problemei

**Context:** magazinul a trimis o campanie de marketing. Coloana `Response` = 1 dacă clientul **a acceptat oferta** din ultima campanie, 0 altfel.

```python
df = pd.read_csv("data/raw.csv")
print("Dimensiune:", df.shape)
df.head()
```


## 2. Analiza exploratorie a datelor (EDA)

Înainte de orice model mă uit la date: tipuri de coloane, valori lipsă, valori ciudate, distribuția țintei.

```python
df.info()
```

```python
print("Valori lipsă pe coloană:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nColoane cu o singură valoare (nu aduc informație):",
      [c for c in df.columns if df[c].nunique() == 1])
```

**Observații:**
- doar `Income` are valori lipsă (24 din 2240, ~1%) → le vom completa (imputare) **în pipeline**;
- `Z_CostContact` și `Z_Revenue` sunt constante → le eliminăm;
- `ID` e doar un identificator → îl eliminăm (altfel modelul ar putea „învăța” numere fără sens);
- `Education` și `Marital_Status` sunt text → trebuie codificate.


## 2.1 Distribuția țintei – sunt clasele echilibrate?

```python
counts = df["Response"].value_counts()
print(counts)
print("\nProcent:", (df["Response"].value_counts(normalize=True) * 100).round(1).to_dict())

counts.plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Response: 0 = nu a acceptat, 1 = a acceptat")
plt.xticks(rotation=0); plt.ylabel("nr. clienți"); plt.show()
```

**Doar ~15% dintre clienți au răspuns** → clase **dezechilibrate** (≈ 1 la 6).


## 2.2 Variabile numerice – valori aberante

```python
df[["Year_Birth", "Income", "Recency", "MntWines", "MntMeatProducts", "NumWebVisitsMonth"]].describe().round(1)
```

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
axes[0].hist(df["Year_Birth"], bins=40); axes[0].set_title("Year_Birth")
axes[1].hist(df["Income"].dropna(), bins=40); axes[1].set_title("Income")
axes[2].hist(df["MntWines"], bins=40); axes[2].set_title("MntWines (cheltuieli pe vin)")
plt.tight_layout(); plt.show()

print("Născuți înainte de 1920:", df.loc[df["Year_Birth"] < 1920, "Year_Birth"].tolist())
print("Venituri > 200.000:", df.loc[df["Income"] > 200_000, "Income"].tolist())
```

**Observații:**
- există clienți „născuți” în 1893–1900 (vârstă > 110 ani) → **erori de introducere**;
- un venit de 666.666 → tot eroare evidentă;
- cheltuielile sunt foarte asimetrice (mulți cheltuie puțin, puțini cheltuie mult) – normal pentru date de vânzări.


## 2.3 Cine răspunde? Rata de răspuns pe grupe

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
df.groupby("Education")["Response"].mean().sort_values().plot(kind="barh", ax=axes[0], title="Rata de răspuns pe educație")
df.groupby("Marital_Status")["Response"].mean().sort_values().plot(kind="barh", ax=axes[1], title="Rata de răspuns pe stare civilă")
camp = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
df.groupby(df[camp].sum(axis=1))["Response"].mean().plot(kind="bar", ax=axes[2],
        title="Rata de răspuns vs nr. campanii acceptate înainte")
axes[2].set_xlabel("campanii anterioare acceptate"); axes[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()
```

```python
# corelația fiecărei coloane numerice cu ținta
num = df.select_dtypes("number").drop(columns=["ID", "Z_CostContact", "Z_Revenue"])
corr = num.corr()["Response"].drop("Response").sort_values()
corr.plot(kind="barh", figsize=(7, 7), title="Corelația cu Response")
plt.axvline(0, color="k", lw=0.8); plt.show()
```

**Ce am aflat din EDA:**
- clienții care **au acceptat campanii anterioare** răspund mult mai des (cel mai puternic semnal);
- cei care cumpără mult (vin, carne, din catalog) și au venit mare răspund mai des;
- `Recency` (zile de la ultima cumpărare) e corelat **negativ**: clienții activi recent răspund mai des;
- clienții cu copii acasă (`Kidhome`, `Teenhome`) răspund mai rar;
- `Single`/`Divorced`/`Widow` răspund mai des decât `Married`/`Together`;
- categoriile `Alone`, `Absurd`, `YOLO` au 2–3 persoane → le grupez în `Other` (prea rare ca modelul să învețe ceva din ele).


## 3. Curățare și atribute noi

Aici fac doar transformări **rând cu rând** (care nu folosesc statistici din tot setul) → sigure din punct de vedere al data leakage, pot fi făcute înainte de împărțirea train/test.

```python
data = df.copy()

# 1) eliminăm rândurile imposibile (vârstă > 100, venit absurd)
data = data[(data["Year_Birth"] >= 1920) & ~(data["Income"] > 200_000)]

# 2) atribute noi
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
data["Age"] = 2014 - data["Year_Birth"]
data["Customer_Days"] = (pd.Timestamp("2014-12-31") - data["Dt_Customer"]).dt.days
data["Children"] = data["Kidhome"] + data["Teenhome"]
mnt = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
data["MntTotal"] = data[mnt].sum(axis=1)
data["TotalPurchases"] = data[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
data["PrevAccepted"] = data[camp].sum(axis=1)

# 3) categorii rare -> "Other"
data["Marital_Status"] = data["Marital_Status"].replace({"Alone": "Other", "Absurd": "Other", "YOLO": "Other"})

# 4) eliminăm coloanele inutile
data = data.drop(columns=["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"])

print("Rânduri eliminate:", len(df) - len(data))
print("Dimensiune finală:", data.shape)
print("Income lipsă (rămâne pentru pipeline):", data["Income"].isna().sum())
```


## 4. Împărțirea train/test și pipeline-ul de preprocesare

- **80% train, 20% test**, cu `stratify=y` → ambele părți păstrează aceeași proporție de 15% clienți care răspund.
- Setul de **test e pus deoparte și folosit O SINGURĂ DATĂ, la final**. Toate deciziile (model, hiperparametri) se iau doar pe train, cu cross-validation.

```python
X = data.drop(columns="Response")
y = data["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

print("Train:", X_train.shape, "| % pozitive:", round(y_train.mean() * 100, 1))
print("Test :", X_test.shape,  "| % pozitive:", round(y_test.mean() * 100, 1))
```


## 4.2 Pipeline-ul de preprocesare

```python
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("Numerice  :", len(num_cols), num_cols)
print("Categoriale:", cat_cols)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
preprocess
```

```python
# verificare: după preprocesare nu mai există valori lipsă și totul e numeric
Xt = preprocess.fit_transform(X_train)
print("Formă după preprocesare:", Xt.shape, "| NaN rămase:", np.isnan(Xt).sum())
```


## Exemplu: un client real, înainte și după preprocesare

Iau primul client din train și arăt ce face pipeline-ul cu el:
- **numericele** (`Income`, `Recency`, `Kidhome`) devin „cât de departe e clientul de cel obișnuit”: 0 = ca media, plus = peste medie, minus = sub medie;
- **textul** (`Education`, `Marital_Status`) devine coloane 0/1 (**One-Hot**): pune 1 la categoria clientului și 0 la toate celelalte.

```python
from IPython.display import HTML, display

client = X_train.iloc[0]                                     # primul client din train
dupa = pd.Series(preprocess.transform(X_train.iloc[[0]])[0],  # același client, după pipeline
                 index=[n.split("__")[1] for n in preprocess.get_feature_names_out()])

# rândurile colorate = aceleași informații, înainte și după
colorate_inainte = ["Education", "Marital_Status", "Income", "Kidhome", "Recency"]
colorate_dupa = ["Income", "Kidhome", "Recency", "Education_Graduation", "Marital_Status_Divorced"]

def scrie_inainte(nume, v):
    return f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)

def scrie_dupa(nume, v):
    if nume.startswith(("Education_", "Marital_Status_")):
        return str(int(v))                                   # One-Hot: 0 sau 1
    return f"{v:.3f}".replace(".", ",")                     # numerice scalate

def tabel(serie, titlu, colorate, scrie):
    randuri = ""
    for nume, v in serie.items():
        fundal = "background: rgba(11, 122, 87, 0.25);" if nume in colorate else ""
        randuri += (f"<tr style='{fundal}'><td style='text-align:left; padding:4px 10px'>{nume}</td>"
                    f"<td style='text-align:right; padding:4px 10px; font-family:monospace'>{scrie(nume, v)}</td></tr>")
    return (f"<div style='flex:1; min-width:260px'><p style='font-family:monospace; letter-spacing:1px'>{titlu}</p>"
            f"<table style='width:100%; border-collapse:collapse'>{randuri}</table></div>")

display(HTML("<p><b>Un client real din train, înainte și după pipeline</b><br>"
             "Rândurile colorate sunt aceleași informații, transformate. "
             "Textul a devenit coloane 0/1, iar numerele au fost scalate.</p>"
             "<div style='display:flex; gap:32px; flex-wrap:wrap'>"
             + tabel(client, f"ÎNAINTE · {len(client)} DE COLOANE", colorate_inainte, scrie_inainte)
             + tabel(dupa, f"DUPĂ · {len(dupa)} DE COLOANE", colorate_dupa, scrie_dupa)
             + "</div>"))
```

**Cum se citește:**
- `Income` 57.136 → **0,244**: puțin peste venitul mediu din train (51.948);
- `Recency` 18 zile → **−1,071**: a cumpărat mult mai recent decât clientul obișnuit (49 de zile);
- `Kidhome` 0 → **−0,829**: mai puțini copii mici decât media (0,45);
- `Education` = Graduation → `Education_Graduation = 1`, celelalte 4 coloane de educație = 0;
- `Marital_Status` = Divorced → `Marital_Status_Divorced = 1`, celelalte 5 = 0.


## 4.3 Cum evit **data leakage**

*Data leakage* = informație din datele de test ajunge (direct sau indirect) în antrenare → scoruri prea optimiste, care nu se mai confirmă în realitate.


## 4.4 Cross-validation

Folosesc **`StratifiedKFold` cu k = 5**: train-ul e împărțit în 5 bucăți; de 5 ori antrenez pe 4 și validez pe a 5-a, apoi fac media.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```


## 5. Tratarea dezechilibrului dintre clase

Am 85% / 15%. Un model antrenat „normal” tinde să favorizeze clasa majoritară (spune des „nu”) și ratează clienții care ar răspunde – exact cei care ne interesează.

```python
variante = {
    "Fără tratare":      Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=2000))]),
    "class_weight":      Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "SMOTE":             ImbPipeline([("prep", preprocess), ("smote", SMOTE(random_state=SEED)),
                                      ("model", LogisticRegression(max_iter=2000))]),
}

rez = []
for nume, pipe in variante.items():
    r = {"Variantă": nume}
    for m in ["roc_auc", "f1", "recall", "precision"]:
        r[m] = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=m).mean()
    rez.append(r)
pd.DataFrame(rez).set_index("Variantă").round(3)
```

**Interpretare:**
- **ROC-AUC rămâne aproape la fel** – capacitatea modelului de a *ordona* clienții (cine e mai probabil să răspundă) nu se schimbă;
- ce se schimbă e **pragul de decizie**: fără tratare modelul găsește puțini dintre cei care răspund (recall mic); cu `class_weight` sau SMOTE **recall-ul crește mult**, cu prețul unei precizii mai mici (mai multe oferte trimise „degeaba”);
- pentru o campanie de marketing, a rata un client interesat costă mai mult decât a trimite un email în plus → prefer **recall mai mare**.


## 6. Modelele și alegerea hiperparametrilor (Grid Search + 5-fold CV)

```python
modele = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}),
    "SVM": (
        SVC(class_weight="balanced", random_state=SEED),
        [{"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100], "model__gamma": [0.001, 0.01, 0.1, 1]},
         {"model__kernel": ["linear"], "model__C": [0.1, 1]}]),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"model__n_estimators": [300], "model__max_depth": [3, 5, 10, None],
         "model__min_samples_leaf": [1, 5, 10]}),
}

cele_mai_bune = {}
cautari = {}      # păstrăm toate căutările pentru studiile de senzitivitate (cap. 10)
rezumat_cv = []
for nume, (model, grid) in modele.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train)
    cele_mai_bune[nume] = gs.best_estimator_
    cautari[nume] = gs
    i = gs.best_index_
    rezumat_cv.append({"Model": nume,
                       "ROC-AUC CV (medie)": gs.cv_results_["mean_test_score"][i],
                       "± std": gs.cv_results_["std_test_score"][i],
                       "Cei mai buni parametri": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
    print(f"{nume:20s} ROC-AUC CV = {gs.best_score_:.3f}  | {gs.best_params_}")

# baseline Dummy
dummy = Pipeline([("prep", preprocess), ("model", DummyClassifier(strategy="stratified", random_state=SEED))])
s = cross_val_score(dummy, X_train, y_train, cv=cv, scoring="roc_auc")
cele_mai_bune["Dummy (baseline)"] = dummy.fit(X_train, y_train)
rezumat_cv.append({"Model": "Dummy (baseline)", "ROC-AUC CV (medie)": s.mean(), "± std": s.std(), "Cei mai buni parametri": "-"})

tabel_cv = pd.DataFrame(rezumat_cv).set_index("Model").sort_values("ROC-AUC CV (medie)", ascending=False)
tabel_cv.round(3)
```

```python
t = tabel_cv.sort_values("ROC-AUC CV (medie)")
plt.barh(t.index, t["ROC-AUC CV (medie)"], xerr=t["± std"], color="#4C72B0", capsize=4)
plt.axvline(0.5, ls="--", c="gray"); plt.xlim(0.4, 1)
plt.title("ROC-AUC în 5-fold CV (pe train), cu deviația standard"); plt.show()
```

**Interpretare:**
- toate cele 3 modele sunt **mult peste Dummy (≈0.5 = ghicit la întâmplare)** → chiar au învățat ceva;
- diferențele dintre modele sunt mici și comparabile cu deviația standard dintre fold-uri → niciun model nu e „net superior”;
- Regresia Logistică ține pasul cu modelele complexe → relația dintre atribute și răspuns e în mare parte **aproape liniară** (venit, cheltuieli, campanii anterioare → mai multe șanse).


## 7. Evaluarea finală pe setul de test

Abia acum folosesc setul de test – **o singură dată**, pentru modelele deja alese.

```python
def scor_continuu(model, X):
    # SVC nu are predict_proba (fără probability=True), dar decision_function e suficient pentru AUC
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

rezultate_test = []
for nume, model in cele_mai_bune.items():
    y_pred = model.predict(X_test)
    s = scor_continuu(model, X_test)
    rezultate_test.append({"Model": nume,
                           "ROC-AUC": roc_auc_score(y_test, s),
                           "PR-AUC": average_precision_score(y_test, s),
                           "Recall": recall_score(y_test, y_pred),
                           "Precision": precision_score(y_test, y_pred, zero_division=0),
                           "F1": f1_score(y_test, y_pred),
                           "Accuracy": accuracy_score(y_test, y_pred)})
tabel_test = pd.DataFrame(rezultate_test).set_index("Model").sort_values("ROC-AUC", ascending=False)
tabel_test.round(3)
```

```python
fig, ax = plt.subplots(figsize=(6, 5))
for nume, model in cele_mai_bune.items():
    RocCurveDisplay.from_predictions(y_test, scor_continuu(model, X_test), name=nume, ax=ax)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_title("Curbe ROC pe setul de test"); plt.show()
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, nume in zip(axes, ["Logistic Regression", "SVM", "Random Forest"]):
    ConfusionMatrixDisplay.from_estimator(cele_mai_bune[nume], X_test, y_test, ax=ax, colorbar=False,
                                          display_labels=["nu răspunde", "răspunde"])
    ax.set_title(nume)
plt.tight_layout(); plt.show()
```

**Cum citesc matricea de confuzie:** pe diagonală sunt predicțiile corecte. Colțul **stânga-jos** = clienți care ar fi răspuns, dar modelul i-a ratat (*false negatives*) – vrem cât mai puțini. Colțul **dreapta-sus** = oferte trimise degeaba (*false positives*) – mai puțin grav (costul unui email).


## 8. Comparația modelelor

```python
cmp = tabel_cv[["ROC-AUC CV (medie)"]].join(tabel_test[["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]])
cmp.columns = ["ROC-AUC (CV train)", "ROC-AUC (test)", "PR-AUC (test)", "Recall (test)", "Precision (test)", "F1 (test)"]
cmp.round(3)
```

```python
cmp.drop(index="Dummy (baseline)")[["ROC-AUC (test)", "PR-AUC (test)", "Recall (test)", "F1 (test)"]].plot(
    kind="bar", figsize=(9, 4), rot=0, ylim=(0, 1), title="Comparația modelelor pe test")
plt.legend(loc="lower right"); plt.show()
```


## 9. Interpretarea rezultatelor – ce atribute contează?

Folosesc **permutation importance** pe setul de test: amestec aleator valorile unei coloane și văd cât scade ROC-AUC. Dacă scade mult → modelul se bazează pe acea coloană.

```python
nume_model = tabel_test.drop(index="Dummy (baseline)").index[0]   # cel mai bun model pe test
print("Model analizat:", nume_model)
pi = permutation_importance(cele_mai_bune[nume_model], X_test, y_test, scoring="roc_auc",
                            n_repeats=20, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
imp.tail(12).plot(kind="barh", figsize=(7, 5), xerr=pd.Series(pi.importances_std, index=X_test.columns)[imp.tail(12).index],
                  title=f"Top 12 atribute – permutation importance ({nume_model})")
plt.xlabel("scăderea ROC-AUC când coloana e amestecată"); plt.show()
print(imp.sort_values(ascending=False).head(8).round(3))
```

```python
# Sensul efectului: coeficienții Regresiei Logistice (atributele sunt scalate, deci comparabile)
lr = cele_mai_bune["Logistic Regression"]
nume_coloane = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["model"].coef_[0], index=nume_coloane).sort_values()
coef.index = coef.index.str.replace("num__", "").str.replace("cat__", "")
pd.concat([coef.head(6), coef.tail(6)]).plot(kind="barh", figsize=(7, 5),
        color=["#C44E52"] * 6 + ["#55A868"] * 6, title="Regresia Logistică: verde = crește șansa, roșu = o scade")
plt.axvline(0, c="k", lw=0.8); plt.show()
```


## 10. Cât de sigure sunt rezultatele?

Din grid search am păstrat scorul **fiecărei** combinații (pe train și pe validare). Așa văd nu doar „care e cel mai bun”, ci și **cât de mult contează** fiecare hiperparametru și unde apare overfitting-ul (scor mare pe train, mic pe validare).

```python
def rezultate(nume):
    r = pd.DataFrame(cautari[nume].cv_results_)
    r.columns = [c.replace("param_model__", "") for c in r.columns]
    return r

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

# (a) Regresia Logistică: ROC-AUC în funcție de C
r = rezultate("Logistic Regression").sort_values("C")
axes[0].plot(r["C"], r["mean_train_score"], "o--", label="train")
axes[0].plot(r["C"], r["mean_test_score"], "o-", label="validare (CV)")
axes[0].fill_between(r["C"].astype(float), r["mean_test_score"] - r["std_test_score"],
                     r["mean_test_score"] + r["std_test_score"], alpha=0.2)
axes[0].set_xscale("log"); axes[0].set_xlabel("C"); axes[0].set_ylabel("ROC-AUC")
axes[0].set_title("Regresia Logistică: efectul lui C"); axes[0].legend()

# (b) SVM RBF: hartă C x gamma
r = rezultate("SVM"); r = r[r["kernel"] == "rbf"]
h = r.pivot_table(index="C", columns="gamma", values="mean_test_score")
im = axes[1].imshow(h.values, cmap="viridis", aspect="auto")
axes[1].set_xticks(range(len(h.columns)), h.columns); axes[1].set_yticks(range(len(h.index)), h.index)
axes[1].set_xlabel("gamma"); axes[1].set_ylabel("C"); axes[1].set_title("SVM (RBF): ROC-AUC pe validare")
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        axes[1].text(j, i, f"{h.values[i, j]:.3f}", ha="center", va="center",
                     color="white" if h.values[i, j] < h.values.max() - 0.05 else "black")
plt.colorbar(im, ax=axes[1])

# (c) Random Forest: adâncimea arborilor
r = rezultate("Random Forest")
r["depth"] = r["max_depth"].astype(object).where(r["max_depth"].notna(), "None").astype(str)
ordine = ["3", "5", "10", "None"]
for leaf, g in r.groupby("min_samples_leaf"):
    g = g.set_index("depth").loc[ordine]
    axes[2].plot(ordine, g["mean_test_score"], "o-", label=f"validare, leaf={leaf}")
    axes[2].plot(ordine, g["mean_train_score"], "x--", alpha=0.5, label=f"train, leaf={leaf}")
axes[2].set_xlabel("max_depth"); axes[2].set_title("Random Forest: adâncime și min_samples_leaf")
axes[2].legend(fontsize=7)
plt.tight_layout(); plt.show()

print("SVM (RBF) – ROC-AUC pe validare:"); print(h.round(3))
print(); print("Random Forest – ROC-AUC validare / train:")
print(r.pivot_table(index="min_samples_leaf", columns="depth", values=["mean_test_score", "mean_train_score"]).round(3))
print(); print("Regresia Logistică:"); print(rezultate("Logistic Regression")[["C", "mean_train_score", "mean_test_score", "std_test_score"]].round(3))
# rezultatele SVM liniar, pentru comparație
print(); print("SVM liniar:"); print(rezultate("SVM").query("kernel == 'linear'")[["C", "mean_test_score", "std_test_score"]].round(3))
```


## 10.2 Mai multe rulări: cât de stabile sunt scorurile?

Rezultatele de pe test depind de **cum a căzut împărțirea** train/test (test mic: ~67 de clienți care răspund → un singur client în plus sau în minus schimbă recall-ul cu ~1.5%). Repet **împărțirea + antrenarea de 10 ori, cu 10 seed-uri diferite**, păstrând hiperparametrii aleși, și raportez **media ± deviația standard**.

```python
from sklearn.base import clone

rulari = []
for seed in range(10):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    for nume in ["Logistic Regression", "SVM", "Random Forest"]:
        m = clone(cele_mai_bune[nume])
        if "random_state" in m.named_steps["model"].get_params():
            m.set_params(model__random_state=seed)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rulari.append({"seed": seed, "Model": nume, "ROC-AUC": roc_auc_score(yte, scor_continuu(m, Xte)),
                       "Recall": recall_score(yte, p), "Precision": precision_score(yte, p), "F1": f1_score(yte, p)})
rulari = pd.DataFrame(rulari)
stabilitate = rulari.groupby("Model")[["ROC-AUC", "Recall", "Precision", "F1"]].agg(["mean", "std"]).round(3)
stabilitate
```

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([rulari.loc[rulari["Model"] == n, "ROC-AUC"] for n in ["Logistic Regression", "SVM", "Random Forest"]],
           tick_labels=["Logistic Regression", "SVM", "Random Forest"])
ax.set_ylabel("ROC-AUC pe test"); ax.set_title("ROC-AUC pe 10 împărțiri train/test diferite"); plt.show()
```

**Concluzie:** diferențele dintre modele (câteva miimi de ROC-AUC) sunt **mai mici decât variația de la o împărțire la alta** → SVM și Regresia Logistică sunt **la egalitate** (aceeași medie, aceeași variație). Random Forest are un ROC-AUC mediu asemănător, dar **recall mult mai mic și mai instabil** (deviație standard de 2–3 ori mai mare).


## 10.3 Contează preprocesarea? (experimente „cu / fără”)

Scot pe rând câte o componentă și măsor efectul în **5-fold CV pe train** (testul rămâne neatins):
- SVM **fără scalare**;
- SVM **fără `class_weight`**;
- Regresia Logistică **fără atributele noi** (`Children`, `MntTotal`, `TotalPurchases`, `PrevAccepted`).

```python
from sklearn.model_selection import cross_validate

def evalueaza_cv(pipe, X_, nume):
    r = cross_validate(pipe, X_, y_train, cv=cv, scoring=["roc_auc", "recall", "f1"])
    return {"Experiment": nume, "ROC-AUC": r["test_roc_auc"].mean(), "Recall": r["test_recall"].mean(), "F1": r["test_f1"].mean()}

svm_best = cele_mai_bune["SVM"]
fara_scalare = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])
noi = ["Children", "MntTotal", "TotalPurchases", "PrevAccepted"]
prep_fara_noi = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
     [c for c in num_cols if c not in noi]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])

ablatie = pd.DataFrame([
    evalueaza_cv(svm_best, X_train, "SVM – complet (ales)"),
    evalueaza_cv(Pipeline([("prep", fara_scalare), ("model", clone(svm_best.named_steps["model"]))]), X_train, "SVM – fără scalare"),
    evalueaza_cv(clone(svm_best).set_params(model__class_weight=None), X_train, "SVM – fără class_weight"),
    evalueaza_cv(cele_mai_bune["Logistic Regression"], X_train, "LogReg – complet (ales)"),
    evalueaza_cv(Pipeline([("prep", prep_fara_noi), ("model", clone(cele_mai_bune["Logistic Regression"].named_steps["model"]))]),
                 X_train.drop(columns=noi), "LogReg – fără atributele noi"),
]).set_index("Experiment")
ablatie.round(3)
```

**Concluzie:**
- **Scalarea e esențială pentru SVM**: fără ea, `Income` (zeci de mii) domină distanțele și modelul aproape nu mai învață;
- **`class_weight` nu schimbă ROC-AUC, dar dublează aproape recall-ul** – exact efectul văzut în secțiunea 5;
- **atributele noi** aduc un câștig mic: informația exista deja în coloanele originale, atributele doar o rezumă.


## 10.4 Verificare: clienți duplicați

Setul are clienți care apar de mai multe ori (rânduri identice, cu `ID` diferit). Dacă același client e și în train și în test, scorul pe test poate fi puțin optimist (un fel de *leakage*). Verific câte rânduri din test au un „geamăn” în train și reevaluez modelul ales **fără ele**.

```python
cheie = lambda d: pd.util.hash_pandas_object(d, index=False)   # o „amprentă” pentru fiecare rând
in_train = cheie(X_test).isin(set(cheie(X_train)))
print(f"Rânduri din test care apar identic și în train: {in_train.sum()} din {len(X_test)}")

m = cele_mai_bune["SVM"]
for nume, masca in [("Test complet", np.ones(len(X_test), bool)), ("Test fără duplicate", ~in_train.values)]:
    p = m.predict(X_test[masca])
    print(f"{nume:22s} ROC-AUC = {roc_auc_score(y_test[masca], scor_continuu(m, X_test[masca])):.3f} | "
          f"Recall = {recall_score(y_test[masca], p):.3f} | n = {masca.sum()}")
```

**Concluzie:** ~16% din rândurile de test au un duplicat în train. Fără ele, scorul **nu scade** (ba chiar crește puțin) → duplicatele **nu umflă artificial** rezultatele; modelul nu le „recunoaște”, ci generalizează.


## 11. A doua întrebare: cât cheltuie un client? (regresie)

Povestea setului pune două întrebări: *cine acceptă oferta* (rezolvată mai sus) și *cât cheltuie un client*. Ținta este `MntTotal` (suma celor 6 coloane `Mnt*`, pe 2 ani).

```python
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

scoase = mnt + ["MntTotal"]
Xr_train, Xr_test = X_train.drop(columns=scoase), X_test.drop(columns=scoase)
yr_train, yr_test = X_train["MntTotal"], X_test["MntTotal"]
num_r = [c for c in num_cols if c not in scoase]

prep_r = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_r),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])
cv_r = KFold(n_splits=5, shuffle=True, random_state=SEED)

modele_r = {
    "Ridge": (Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
    "SVR": (SVR(), {"model__C": [100, 1000, 3000], "model__gamma": [0.01, 0.03, "scale"]}),
    "Random Forest Regressor": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                                {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5]}),
}
reg_rez, cele_mai_bune_r = [], {}
for nume, (model, grid) in modele_r.items():
    gs = GridSearchCV(Pipeline([("prep", prep_r), ("model", model)]), grid, cv=cv_r,
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(Xr_train, yr_train)
    cele_mai_bune_r[nume] = gs.best_estimator_
    reg_rez.append({"Model": nume, "MAE CV": -gs.best_score_,
                    "Parametri": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
cele_mai_bune_r["Dummy (media)"] = Pipeline([("prep", prep_r), ("model", DummyRegressor())]).fit(Xr_train, yr_train)
reg_rez.append({"Model": "Dummy (media)",
                "MAE CV": -cross_val_score(cele_mai_bune_r["Dummy (media)"], Xr_train, yr_train, cv=cv_r,
                                           scoring="neg_mean_absolute_error").mean(), "Parametri": "-"})

for r in reg_rez:
    p = cele_mai_bune_r[r["Model"]].predict(Xr_test)
    r.update({"MAE test": mean_absolute_error(yr_test, p),
              "RMSE test": mean_squared_error(yr_test, p) ** 0.5,
              "R² test": r2_score(yr_test, p)})
tabel_reg = pd.DataFrame(reg_rez).set_index("Model").sort_values("MAE test")
print("Cheltuiala medie a unui client (test):", round(yr_test.mean(), 1))
tabel_reg.round(3)
```

```python
best_r = tabel_reg.index[0]
p = cele_mai_bune_r[best_r].predict(Xr_test)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(yr_test, p, alpha=0.4, s=12)
ax.plot([0, yr_test.max()], [0, yr_test.max()], "k--", lw=0.8)
ax.set_xlabel("cheltuială reală"); ax.set_ylabel("cheltuială prezisă")
ax.set_title(f"{best_r}: real vs prezis (test)"); plt.show()

# ce atribute contează pentru cheltuială?
pi_r = permutation_importance(cele_mai_bune_r[best_r], Xr_test, yr_test, scoring="neg_mean_absolute_error",
                              n_repeats=10, random_state=SEED, n_jobs=-1)
print("Top atribute (creșterea MAE când sunt amestecate):")
print(pd.Series(pi_r.importances_mean, index=Xr_test.columns).sort_values(ascending=False).head(6).round(1))
```


## 12. Concluzii

**Întrebarea 1 – Cine va accepta oferta? Da, se poate prezice destul de bine.** Am comparat Regresia Logistică, SVM și Random Forest (plus un Dummy). Am ales **Regresia Logistică (C = 1, `class_weight="balanced"`)**: ROC-AUC 0.89 pe test (0.90 în CV), găsește ~76–79% din clienții care răspund. SVM (RBF, C = 10, gamma = 0.001) a obținut aceleași scoruri, dar e mai complicat și greu de explicat; Random Forest are precizie mai mare, dar ratează jumătate din clienții interesați.
