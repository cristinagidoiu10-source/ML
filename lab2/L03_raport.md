# Raport L03 — Preprocesarea în pipeline, class imbalance și evaluarea modelelor (T8 Customer Personality Analysis)

Analiză pe fiecare TODO din `L03_Pipeline_CV_Scaling_(1).ipynb`. Pentru fiecare: **verdict**, **ce e greșit și de ce**, **codul de înlocuit** și **codul corect** (copiezi toată celula).

Legendă: ✅ corect · ⚠️ merge, dar are probleme de logică / curățenie · ❌ greșit sau necompletat

> **Important:** L03 pornește de la `clean.csv` din L02. Dacă aplici corecturile din `L02_raport.md` (`666666` → NaN, `2n Cycle` → Master, dată fixă), `clean.csv` are 2058 de rânduri și **toate cifrele de mai jos se schimbă puțin**. Cifrele din acest raport le-am obținut rulând exact codul de aici pe `clean.csv` corectat (scikit-learn 1.9); pe Colab pot diferi la a treia zecimală. Copiază în observații cifrele care îți ies ție.

## Rezumat

| Capitol | TODO | Verdict | Problema principală |
|---|---|---|---|
| 0 | Setul de date | ⚠️ | `MY_NAME` diferă de L02 (spațiul de la final); lipsește ținta |
| 1 | Citire + split | ⚠️ | fallback care dublează curățarea; `digit_cats` nu face nimic; `MntTotal` în X |
| 2.5 | Preprocesarea | ⚠️ | importuri lipsă; ordinea `2n Cycle` greșită; o afirmație despre `Income` e falsă |
| 2.5 | Decizii de preprocesare | ❌ | necompletată |
| 2.6 | Dezechilibrul claselor | ✅ | corect |
| 2.6 | Observații dezechilibru | ⚠️ | motivul alegerii lipsește; nu compară cu SMOTE |
| 3 | Validarea încrucișată | ✅ | corect |
| 3 | Observații CV | ⚠️ | nu răspunde la întrebarea despre abaterea standard |
| 3.3 | Curbe | ✅ | corect |
| 3.3 | Observații curbe | ❌ | „aleg C=1 pentru că e implicit” nu e un argument; curba de învățare nu e discutată |
| 4 | GridSearch | ⚠️ | comentariu cu cifre vechi; nested CV evaluează altă grilă decât cea căutată |
| 4 | Observații GridSearch | ❌ | câștigul pentru random forest e calculat greșit (+0,017 în loc de +0,107) |
| 5 | Test final | ⚠️ | nested CV rulat din nou; construcția tabelului e greoaie |
| 5 | Observații test | ⚠️ | „+0,115” fără să spui din ce vine câștigul |
| 5.2 | Prag (opțional) | — | nefăcut; codul e mai jos |
| — | Celule AI | ⚠️ | „Rezultat” gol peste tot |

---

## TODO 0 — Setul de date (celula 5) ⚠️

### Ce e greșit
1. În L02, `MY_NAME = "Customer Personality Analysis "` (**cu spațiu**). Aici e **fără**. Laboratorul cere *„`MY_NAME` trebuie să fie identic”*: altfel L03 caută `clean.csv` în alt folder decât cel în care l-a salvat L02. La tine a mers (`exists: True`), probabil pentru că ai rulat L02 și cu numele fără spațiu, dar fișierele de rezultate din L02 au spațiu în nume. Soluția: **fără spațiu în ambele** (vezi raportul L02, TODO 0).
2. TODO-ul spune *„completați aceleași valori ca în laboratorul 2”*. Lipsesc ținta și tipul problemei, care apar apoi scrise de mână în celula 8.

### Cod de înlocuit
```python
MY_NAME = "Customer Personality Analysis"            # ex. "penguins"
```

### Cod corect
```python
# TODO: aceleasi valori ca in laboratorul 2
MY_NAME = "Customer Personality Analysis"     # IDENTIC cu L02 (fara spatiu la final)
TARGET = "Response"                           # 1 = a acceptat oferta din ultima campanie, 0 = nu
TASK = "classification"
TARGET_R = "MntTotal"                         # tinta de regresie din L02: nu e atribut la clasificare

MY_DIR = f"{DATA_DIR}/{MY_NAME}"
print(MY_NAME, MY_DIR, "| clean.csv exists:", os.path.exists(f"{MY_DIR}/clean.csv"))
```

---

## TODO 1 — Citire + split (celula 8) ⚠️

### Ce e greșit
1. **Ramura `else` rescrie curățarea din L02.** Ai acum curățarea în **două locuri**, care deja nu mai coincid: aici citești din `data/customer/customer.csv`, iar TODO 1 din L02 cere `MY_DIR/raw.csv`. Dacă schimbi o decizie în L02, trebuie să ții minte să o schimbi și aici. O singură sursă de adevăr: dacă `clean.csv` lipsește, oprește-te cu un mesaj clar (`assert`). Exact asta recomandă și comentariul din enunț: *„este recomandat să terminați laboratorul 2”*.
2. **`digit_cats = []` + două list comprehension-uri** care nu schimbă nimic: 4 linii de cod mort. Decizia („coloanele 0/1 rămân numerice”) e bună, dar se spune într-un comentariu, nu într-o operație care nu face nimic.
3. **`MntTotal` rămâne în X** (se vede în outputul tău: `num__MntTotal`). Este suma exactă a celor 6 coloane `Mnt*`, care sunt deja în X: o coloană 100% redundantă. La regresia logistică ponderile se împart arbitrar între `MntTotal` și componentele lui, iar interpretarea lor nu mai are sens. Apare și în `make_pipeline` ca atribut. O scoatem.

### Cod de înlocuit
Toată celula 8. Liniile problematice:
```python
else:                                                  # nu exista -> refacem rapid curatarea din laboratorul 2
    df = pd.read_csv(f"{DATA_DIR}/customer/customer.csv")
    ...
X = df.drop(columns=[TARGET])
...
digit_cats = []                                        # nicio coloana de mutat in cat_cols
num_cols = [c for c in num_cols if c not in digit_cats]
cat_cols = cat_cols + digit_cats
```

### Cod corect
```python
# TODO: citire, X / y, split
from sklearn.model_selection import train_test_split

# 1. Setul curatat in laboratorul 2 (o singura sursa: daca lipseste, rulati L02)
CLEAN_CSV = f"{MY_DIR}/clean.csv"
assert os.path.exists(CLEAN_CSV), f"Nu exista {CLEAN_CSV}: rulati intai capitolul 5 din laboratorul 2"
df = pd.read_csv(CLEAN_CSV)
print("loaded:", df.shape)

# 2. X / y; MntTotal iese: e tinta de regresie si suma exacta a coloanelor Mnt*, deja prezente in X
X = df.drop(columns=[TARGET, TARGET_R])
y = df[TARGET]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()      # Education, Marital_Status
# Categorii scrise cu cifre: nu avem. AcceptedCmp1-5 si Complain sunt deja 0/1 (one-hot nu aduce nimic in plus),
# iar Kidhome / Teenhome sunt numaratori (0, 1, 2), cu ordine si distante reale -> raman numerice.

# 3. I - Isolate: 80/20, stratificat (clasa 1 ~15%); datele nu sunt o serie in timp -> impartire aleatoare
#    de aici inainte X_test / y_test NU se mai folosesc pana la capitolul 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 4. Dimensiuni si valori lipsa
print("train:", X_train.shape, "| test:", X_test.shape)
print("numeric:", num_cols)
print("categorical:", cat_cols)
print("missing values in train:\n", X_train.isna().sum()[X_train.isna().sum() > 0])
```

---

## TODO 2.5 — Preprocesarea (celula 22) ⚠️

### Ce e greșit
1. **Importuri lipsă:** folosești `SimpleImputer`, `RobustScaler`, `OrdinalEncoder`, `OneHotEncoder` fără să le imporți în celulă. Merge doar pentru că le-au importat **celulele de exemplu**. Dacă cineva rulează doar celulele TODO (sau tu, într-o sesiune nouă), primești `NameError`.
2. **Ordinea `Education` e greșită:** `["Basic", "2n Cycle", "Graduation", "Master", "PhD"]` pune `2n Cycle` *sub* `Graduation`. În sistemul Bologna, ciclul 1 = licență (`Graduation`), **ciclul 2 = master**, ciclul 3 = doctorat. `2n Cycle` și `Master` sunt același nivel. `OrdinalEncoder` nu poate da aceeași valoare la două categorii, de aceea le-am unit la curățare, în L02. Ordinea corectă devine `Basic < Graduation < Master < PhD`.
3. **Afirmația *„rata de răspuns e asemănătoare cu/fără Income”* este falsă**, și nu ai cod care să o verifice. Pe tot setul, clienții fără `Income` au răspuns în **4,2%** din cazuri, față de **15%** la ceilalți. Pe de altă parte, sunt doar 22 de rânduri în train (un singur client care a răspuns), deci diferența nu e solidă. Corect: verifici cu cod și **lași căutarea (capitolul 4) să decidă** (`prep__num__impute__add_indicator: [False, True]`), în loc să afirmi ceva neverificat.
4. Codul de după `make_pipeline` (lista de parametri `prep__num__...`) e copiat din exemplu și nu ajută la TODO. L-am scos.

Restul e **foarte bun**: funcția `make_pipeline` cu parametri, `handle_unknown` la ambele encodere, `min_frequency` pentru `Other` (4 rânduri), `RobustScaler` justificat prin outlieri.

### Cod de înlocuit
Toată celula 22. Liniile cu probleme:
```python
# - add_indicator: NU implicit - doar ~1% lipsa in Income si rata de raspuns e asemanatoare cu/fara Income
...
EDU_ORDER = [["Basic", "2n Cycle", "Graduation", "Master", "PhD"]]
```

### Cod corect
```python
# TODO: preprocesarea pentru setul nostru
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# verificare pentru add_indicator: lipsa venitului spune ceva despre tinta? (doar pe TRAIN)
no_income = X_train["Income"].isna()
print(f"missing Income in train: {no_income.sum()} rows | response rate: "
      f"without Income = {y_train[no_income].mean():.3f}, with Income = {y_train[~no_income].mean():.3f}")

ORD_COLS = ["Education"]
NOM_COLS = [c for c in cat_cols if c not in ORD_COLS]              # Marital_Status
EDU_ORDER = [["Basic", "Graduation", "Master", "PhD"]]             # "2n Cycle" = Master (unit in L02, la curatare)

def make_pipeline(model, scaler=None, num_strategy="median", add_indicator=False):
    """prep + model; scaler=None -> RobustScaler, scaler="passthrough" -> fara scalare (arbori)."""
    num_steps = Pipeline([("impute", SimpleImputer(strategy=num_strategy, add_indicator=add_indicator)),
                          ("scale", scaler if scaler is not None else RobustScaler())])
    # ordinal, in ordinea data de noi; o categorie necunoscuta -> -1 (fara eroare)
    ord_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=EDU_ORDER,
                                                     handle_unknown="use_encoded_value", unknown_value=-1))])
    # nominal -> one-hot; categoriile cu < 10 aparitii in train ("Other") -> o singura coloana "infrequent"
    nom_steps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist"))])
    prep = ColumnTransformer([("num", num_steps, num_cols),
                              ("ord", ord_steps, ORD_COLS),
                              ("nom", nom_steps, NOM_COLS)])
    return Pipeline([("prep", prep), ("model", model)])

# verificare: pipeline-ul ruleaza pe train
pipe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
feature_names = pipe.named_steps["prep"].get_feature_names_out()
print("fit OK |", len(feature_names), "features after preprocessing")
print(feature_names[-7:])                                           # Education (ordinal) + Marital_Status (one-hot)
```

**De ce docstring în loc de 4 linii de comentarii despre parametri:** în Colab, `make_pipeline?` sau hover pe funcție îți afișează docstring-ul. Comentariile de deasupra funcției nu apar acolo.

---

## TODO 2.5 — Decizii de preprocesare (celula 23) ❌

Celula e necompletată. Răspuns (înlocuiește cifrele din paranteză cu ce îți afișează celula de mai sus):
```markdown
**Decizii de preprocesare:**
- **Numerice → mediana:** `Income` are asimetrie 6,8, iar `Mnt*` și `Num*Purchases` între 1,2 și 2,4; mediana nu este trasă de valorile extreme. Practic, contează doar pentru `Income` (22 de lipsuri în train) și `Year_Birth` (2), deci diferența mean/median e neglijabilă (o confirmă căutarea din capitolul 4).
- **`add_indicator`: implicit nu, dar îl testez în GridSearch.** Clienții fără venit au răspuns rar (4,5% față de 15,3% în train), dar sunt doar 22 de rânduri (un singur client a răspuns), deci diferența poate fi întâmplătoare. Căutarea alege `add_indicator=True`, dar câștigul e mic.
- **Scaler → `RobustScaler`:** avem mulți outlieri (`Income`, `Mnt*`, `NumDealsPurchases`, `NumWebPurchases`); mediana și IQR nu sunt deformate de ei. Pentru arbori: `"passthrough"` (arborii nu au nevoie de scalare).
- **`Education` → ordinal**, cu ordinea `Basic < Graduation < Master < PhD` (`2n Cycle` = ciclul 2 Bologna = master, unit cu `Master` la curățare).
- **`Marital_Status` → one-hot** (nominal, fără ordine), cu `min_frequency=10`: `Other` (4 rânduri) devine o coloană „infrequent”.
- **Coloanele 0/1** (`AcceptedCmp1–5`, `Complain`) și numărătorile `Kidhome`/`Teenhome` rămân numerice.
- **`MntTotal` nu e atribut:** e suma exactă a coloanelor `Mnt*` (redundantă) și ținta de regresie.
```

---

## TODO 2.6 — Dezechilibrul claselor (celula 30) ✅

Codul e **corect**: aceleași fold-uri pentru toate strategiile, reechilibrarea în `ImbPipeline` (deci doar pe partea de antrenare a fiecărui fold), metrici care văd clasa rară. Nimic de schimbat.

Un singur detaliu: comentariul `# ~0.15 si ~5.7` e scris de mână. Dacă datele se schimbă (și se schimbă, după corectura din L02), comentariul rămâne vechi. Ar fi mai bine să-l ștergi, pentru că printul de deasupra afișează deja cifrele.

---

## TODO 2.6 — Observații dezechilibru (celula 31) ⚠️

### Ce e greșit
Ai scris *ce s-a întâmplat* (recall 44%, precizia scade), dar nu *de ce ai ales `class_weight`* în locul oversampling-ului sau SMOTE, care au cifre la fel de bune (la tine chiar puțin mai mari: F1 0,581 față de 0,577). Lipsește argumentul: diferențele sunt mai mici decât abaterea standard între fold-uri, deci la scor sunt la egalitate, iar `class_weight` e cea mai simplă variantă.

### Răspuns corect
```markdown
**Observații:** fără reechilibrare, modelul găsește doar 41% dintre clienții care răspund (recall 0,412; acuratețe echilibrată 0,687). Toate cele trei tehnici urcă recall-ul la 0,75–0,78 și acuratețea echilibrată la ≈ 0,79–0,80:

| strategie | recall | precizie | F1 | bal. acc. |
|---|---|---|---|---|
| none | 0,412 | 0,673 | 0,501 | 0,687 |
| class_weight | 0,784 | 0,431 | 0,555 | 0,798 |
| oversampling | 0,780 | 0,433 | 0,555 | 0,798 |
| SMOTE | 0,748 | 0,441 | 0,553 | 0,788 |

Aleg **`class_weight="balanced"`**: are cel mai bun scor (la egalitate cu oversampling), iar diferențele față de oversampling și SMOTE (≤ 0,01) sunt mult sub abaterea standard între fold-uri (≈ 0,04), deci practic sunt la egalitate. În plus, e cea mai simplă variantă: nu creează rânduri artificiale (SMOTE poate genera clienți nerealiști), nu copiază rânduri (risc de overfitting la oversampling) și nu are nevoie de `imblearn`.

**Prețul:** precizia scade de la 0,673 la 0,431. Modelul găsește aproape de 2 ori mai mulți clienți interesați, dar din cei pe care îi propune, doar 43% chiar acceptă oferta (înainte, 67%). Pentru o campanie de marketing compromisul e bun: un contact inutil costă puțin, un client interesat ratat înseamnă o vânzare pierdută.
```

---

## TODO 3 — Validarea încrucișată (celula 39) ✅

Codul e **corect și complet**: aceleași fold-uri, același pipeline, 3 metrici alese dinainte, scor pe train, boxplot salvat în `FIGURES_DIR`, tabel salvat în `RESULTS_DIR`. Ai adăugat și random forest și gradient boosting, cum sugera enunțul.

Observație de logică (nu e o greșeală de cod): **kNN nu are `class_weight`**, deci e singurul model comparat **fără** reechilibrare. Scorul lui mic (0,53) vine în parte din asta, nu doar din model. Dacă vrei o comparație corectă, poți pune kNN într-un `ImbPipeline` cu `RandomOverSampler`. Nu e cerut, dar menționează-l în observații.

---

## TODO 3 — Observații CV (celula 40) ⚠️

### Ce e greșit
- *„regresia logică”* → **logistică**.
- Nu ai răspuns la întrebarea din mijloc: *„Diferențele sunt mai mari decât abaterea standard?”*
- Nu ai observat ce e cel mai interesant în tabel: random forest și gradient boosting au **ROC AUC aproape egal** cu regresia logistică, dar acuratețe echilibrată mult mai mică.

### Răspuns corect (cifrele mele; verifică-le cu tabelul tău)
```markdown
**Observații:**
- **Cea mai bună medie pe validare:** regresia logistică: acuratețe echilibrată 0,798 ± 0,039, F1 0,555, ROC AUC 0,884.
- **Diferențele față de abatere:** față de următorul model (random forest, 0,741 ± 0,035), diferența este 0,057, de aproximativ 1,5 ori abaterea standard. Cutiile din boxplot se suprapun foarte puțin, deci diferența e probabil reală, nu noroc. În schimb, random forest (0,741), gradient boosting (0,718) și arborele cu `max_depth=5` (0,677 ± 0,034) se suprapun parțial.
- **ROC AUC spune altceva:** random forest (0,870) și gradient boosting (0,880) **ordonează** clienții aproape la fel de bine ca regresia logistică (0,884). Le scade acuratețea echilibrată pentru că pragul implicit 0,5 nu li se potrivește: probabilitățile lor sunt adunate spre 0 (arbori care au memorat train-ul). Cu un prag ales (capitolul 5.2), diferența s-ar micșora.
- **Cea mai mare diferență train – validare:** arborele fără limită (train 0,998 / validare 0,610) și gradient boosting (0,998 / 0,718): memorează train-ul (overfitting). Random forest are și el 0,984 pe train. Regresia logistică are 0,829 / 0,802, cea mai mică diferență: nu face overfitting.
- **kNN (0,529, F1 0,125)** este singurul model fără reechilibrare (nu are `class_weight`): prezice aproape mereu „0”, deci comparația îl dezavantajează.
```

---

## TODO 3.3 — Curbe de validare și învățare (celula 45) ✅

Codul e **corect**: `C` pe scară logaritmică, `set_xscale("log")`, aceleași fold-uri, banda ± std, figura salvată cu numele cerut, plus un tabel cu valorile ca să le poți cita. Un mic plus: afișează și valorile curbei de învățare, ca să le poți cita în observații. Adaugă la finalul celulei:
```python
display(pd.DataFrame({"train size": sizes, "train": train_lc.mean(axis=1), "validation": val_lc.mean(axis=1)})
        .round(3).set_index("train size"))
```

---

## TODO 3.3 — Observații curbe (celula 46) ❌

### Ce e greșit
*„Aș alege C=1 deoarece are o valoare implicită”* nu e un argument. Faptul că o valoare e implicită nu spune nimic despre setul tău. Întrebarea este **ce arată curba**. Iar a doua parte a întrebării (*„ce spune curba de învățare?”*) nu are răspuns.

Concluzia (C = 1) poate rămâne aceeași, dar motivul trebuie luat din cifre.

### Răspuns corect (cifrele mele; ale tale vor fi apropiate)
```markdown
**Observații:**
- **Curba de validare (`C`):** la C = 0,001 modelul e prea regularizat (train 0,669 / validare 0,665: ambele jos = underfitting). Scorul de validare crește până la C = 1 (0,798), apoi rămâne practic constant (0,802 la C = 10, 0,803 la C = 100). Diferența train – validare este mică peste tot (≈ 0,03), deci nu apare overfitting nici la C mare. Aleg **C = 1**: din C = 1 curba e plată (diferențele de 0,005 sunt mult sub abaterea de ± 0,04), iar dintre valorile cu același scor aleg modelul cel mai regularizat (cel mai simplu). Căutarea din capitolul 4 alege tot C = 1.
- **Curba de învățare:** cu 131 de exemple modelul memorează (train 0,933 / validare 0,695). Pe măsură ce crește setul, curbele se apropie: la 1316 exemple, train 0,829 / validare 0,796. Validarea mai crește puțin la final (0,779 → 0,791 → 0,796), deci mai multe date ar aduce un câștig mic. Distanța rămasă e mică, iar ambele curbe sunt în jur de 0,80–0,83: modelul e limitat mai degrabă de **bias** (e un model liniar), nu de cantitatea de date. Ar ajuta mai mult **atribute noi** sau un model care prinde interacțiuni, nu doar mai multe rânduri.
```

---

## TODO 4 — Alegerea hiperparametrilor (celula 52) ⚠️

### Ce e greșit
1. **Comentariu cu cifre vechi:** *„random forest (0.752, dar train 0.984…)”*. Tabelul tău de la capitolul 3 arată **0,676** pentru random forest. Comentariul a rămas de la o rulare anterioară sau a fost scris de AI. Un comentariu greșit e mai rău decât niciunul. Scrie cifre în comentarii doar dacă le calculează codul.
2. **Nested CV evaluează altă procedură decât cea căutată.** `grid_lr` caută C × `class_weight` × scaler × imputare (40 de combinații), dar `inner_lr` caută doar `C`, cu `class_weight="balanced"` fixat. Nested CV trebuie să estimeze **exact** procedura „caută + antrenează” pe care o folosești. Altfel estimarea nu se aplică modelului ales. Soluția: refolosești același dicționar `param_grid_lr`.
3. **Variabila `nested` e suprascrisă în buclă**, iar în celula de test ai fost nevoit să rulezi din nou nested CV (`nested_lr = cross_val_score(...)`), adică încă 200 de antrenări. Păstrează rezultatele într-un dicționar.
4. Grila de preprocesare: imputarea mean/median nu contează la ~20 de lipsuri (tabelul tău arată scoruri identice), dar `add_indicator` merită testat (vezi TODO 2.5). Am înlocuit-o.

### Cod de înlocuit
Toată celula 52. Liniile cu probleme:
```python
#    regresia logistica (cea mai buna, 0.808) si random forest (0.752, dar train 0.984 -> overfitting, poate fi reglat)
...
    "prep__num__impute__strategy": ["mean", "median"],
...
inner_lr = GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
                        {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=inner_cv, scoring="balanced_accuracy")
...
for name, inner in [("logistic regression", inner_lr), ("random forest", inner_rf)]:
    nested = cross_val_score(inner, X_train, y_train, cv=cv5, scoring="balanced_accuracy")
```

### Cod corect
```python
# TODO: cautarea hiperparametrilor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

SCORING = "balanced_accuracy"          # aleasa INAINTE de cautare: tinta e dezechilibrata

def show_top(grid, title, n=5):
    print(f"=== {title} ===")
    print("best parameters:", grid.best_params_)
    print(f"best CV {SCORING}: {grid.best_score_:.3f}  (slightly optimistic)")
    cols = [c for c in grid.cv_results_ if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_train_score"]
    display(pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[cols].head(n).round(3))

# 1-2a. Regresia logistica: 5 x 2 x 2 x 2 = 40 de combinatii
param_grid_lr = {
    "model__C": [0.01, 0.1, 1, 10, 100],                          # inversul regularizarii, scara logaritmica
    "model__class_weight": [None, "balanced"],                    # verificam din nou alegerea de la 2.6
    "prep__num__scale": [StandardScaler(), RobustScaler()],       # alegere de PREPROCESARE
    "prep__num__impute__add_indicator": [False, True],            # lipsa venitului = informatie? (vezi 2.5)
}
grid_lr = GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000)), param_grid_lr, cv=cv5,
                       scoring=SCORING, n_jobs=-1, return_train_score=True)
grid_lr.fit(X_train, y_train)                                     # 40 combinatii x 5 fold-uri = 200 de antrenari
show_top(grid_lr, "LOGISTIC REGRESSION")

# 1-2b. Random forest: 3 x 4 x 2 = 24 de combinatii (fara scalare)
param_grid_rf = {
    "model__max_depth": [5, 10, None],                            # None = fara limita (memoreaza train-ul)
    "model__min_samples_leaf": [1, 3, 5, 10],                     # frunze mai mari -> mai putin overfitting
    "model__max_features": ["sqrt", 0.5],                         # cate atribute poate alege fiecare impartire
}
grid_rf = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
                                     scaler="passthrough"),
                       param_grid_rf, cv=cv5, scoring=SCORING, n_jobs=-1, return_train_score=True)
grid_rf.fit(X_train, y_train)                                     # 24 combinatii x 5 fold-uri = 120 de antrenari
show_top(grid_rf, "RANDOM FOREST")

# Nested CV: aceeasi GRILA ca mai sus in bucla interioara (3 fold-uri), evaluata pe 5 fold-uri exterioare nevazute
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
nested_scores = {}
for name, grid in [("logistic regression", grid_lr), ("random forest", grid_rf)]:
    inner = GridSearchCV(grid.estimator, grid.param_grid, cv=inner_cv, scoring=SCORING, n_jobs=-1)
    nested_scores[name] = cross_val_score(inner, X_train, y_train, cv=cv5, scoring=SCORING)
    print(f"{name:20s} nested CV estimate: {nested_scores[name].mean():.3f} +/- {nested_scores[name].std():.3f}")
```

**Ce am schimbat și de ce:**
- `show_top` afișează automat toate coloanele `param_*`. Înainte scriai lista de coloane de mână, de două ori, și trebuia s-o actualizezi la fiecare schimbare de grilă.
- `GridSearchCV(grid.estimator, grid.param_grid, ...)` construiește bucla interioară **din aceeași grilă**, deci nested CV estimează exact procedura folosită. Nu mai poți uita un parametru.
- `nested_scores` e un dicționar: celula de test îl folosește direct, fără să ruleze din nou.

---

## TODO 4 — Observații GridSearch (celula 53) ❌

### Ce e greșit
- *„random forest a câștigat cu +0,017”*: **greșit**. În tabelul tău, random forest avea **0,676** la capitolul 3 și **0,783** după căutare: câștigul este **+0,107**, de 6 ori mai mare. Regresia logistică: 0,808 → 0,817 = **+0,009** (sub abaterea standard, deci practic zero).
- Nu ai explicat **ce înseamnă** hiperparametrii aleși (întrebarea cere explicit: *„un arbore mai puțin adânc? un k mare?”*).

### Răspuns corect (cifrele mele; la tine vor fi apropiate)
```markdown
**Observații:**
- **Regresia logistică:** cea mai bună combinație este C = 1, `class_weight="balanced"` (confirmă alegerea de la 2.6), `StandardScaler`, `add_indicator=True`: 0,808 ± 0,033. Primele 5 combinații sunt toate „balanced” și cu `add_indicator=True`, dar diferă între ele cu cel mult 0,004, iar C = 1, 10 și 100 sunt practic egale (curba de validare era plată). Câștig față de capitolul 3: 0,798 → 0,808 = **+0,010**, mai puțin decât abaterea standard (0,033). Setările inițiale erau deja bune; căutarea doar le confirmă.
- **Random forest:** cel mai bun este `min_samples_leaf=10`, `max_features="sqrt"`, `max_depth=None`: frunze cu cel puțin 10 clienți, deci arbori care nu mai pot memora cazuri individuale (adâncimea nu mai contează când frunzele sunt mari). Câștig față de capitolul 3 (`min_samples_leaf=3`): 0,741 → 0,772 = **+0,031**. Scorul pe train a scăzut de la 0,984 la 0,923: overfitting redus, dar nu eliminat.
- **Nested CV:** regresia logistică 0,798 ± 0,038, puțin sub `best_score_` (0,808). Exact asta spune teoria: cea mai bună din 40 de combinații a avut și puțin noroc cu fold-urile. Random forest: 0,753 ± 0,019. Regresia logistică rămâne clar peste, deci **aleg regresia logistică**.
```

---

## TODO 5 — Testul final (celula 60) ⚠️

### Ce e greșit
1. **Rulezi din nou nested CV** (`nested_lr = cross_val_score(inner_lr, ...)`) în celula de test: încă 200 de antrenări, pentru un rezultat pe care îl aveai deja. Folosește `nested_scores` din capitolul 4.
2. **Construcția tabelului final** are 8 linii de `= np.nan` / `.loc[...]`, iar rândul `tuned` are `NaN` la abatere, deși o ai în `grid_lr.cv_results_`.
3. `import joblib` e deja făcut în celula 3; aici e redundant.

Restul e **corect**: o singură evaluare pe test, aceleași metrici ca în CV, comparație cu `best_score_` și nested CV, matricea de confuzie, modelul complet salvat cu `joblib`.

### Cod de înlocuit
Toată celula 60. Liniile cu probleme:
```python
# estimarea nested CV pentru regresia logistica (variabila "nested" din celula anterioara a fost suprascrisa de random forest)
nested_lr = cross_val_score(inner_lr, X_train, y_train, cv=cv5, scoring="balanced_accuracy")
...
cv_results_final = cv_results.copy()
cv_results_final.loc["logistic regression (tuned)"] = np.nan
...
cv_results_final.loc["logistic regression (tuned)", ["test bal. acc.", "test F1", "test ROC AUC"]] = list(test_scores.values())
```

### Cod corect
```python
# TODO: testul final - O SINGURA evaluare pe X_test
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             classification_report, ConfusionMatrixDisplay)

# 1. Modelul ales = best_estimator_ (deja reantrenat pe tot X_train)
best_model = grid_lr.best_estimator_
print("chosen model:", best_model.named_steps["model"])

# 2. Evaluarea pe test, cu aceleasi metrici ca in CV
y_pred_te = best_model.predict(X_test)                       # prag implicit 0.5
y_proba_te = best_model.predict_proba(X_test)[:, 1]          # probabilitatea clasei 1 (pentru ROC AUC)
test_scores = {"bal. acc.": balanced_accuracy_score(y_test, y_pred_te),
               "F1": f1_score(y_test, y_pred_te),
               "ROC AUC": roc_auc_score(y_test, y_proba_te)}

# 3. Testul fata de estimarile din validarea incrucisata (aceeasi metrica)
nested_lr = nested_scores["logistic regression"]
print(f"CV (grid): {grid_lr.best_score_:.3f}  |  nested CV: {nested_lr.mean():.3f} +/- {nested_lr.std():.3f}  |  "
      f"TEST: {test_scores['bal. acc.']:.3f}")
print(f"TEST F1 = {test_scores['F1']:.3f}  |  TEST ROC AUC = {test_scores['ROC AUC']:.3f}")
print(classification_report(y_test, y_pred_te, digits=3))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_te, cmap="Blues")
plt.title("Best logistic regression pipeline - test set")
plt.show()

# 4. Salvam modelul complet (preprocesare + model) si adaugam randul cu scorul de test in tabel
joblib.dump(best_model, f"{MODELS_DIR}/L03_{MY_NAME}_best.joblib")

tuned = "logistic regression (tuned)"
cv_results_final = cv_results.copy()
cv_results_final.loc[tuned, ["val bal. acc. (mean)", "val bal. acc. (std)"]] = [
    grid_lr.best_score_, grid_lr.cv_results_["std_test_score"][grid_lr.best_index_]]
for metric, value in test_scores.items():
    cv_results_final.loc[tuned, f"test {metric}"] = value       # coloana noua: NaN pentru modelele netestate
cv_results_final = cv_results_final.round(3)
display(cv_results_final)
cv_results_final.to_csv(f"{RESULTS_DIR}/L03_{MY_NAME}_cv_results.csv")
# incarcare: model = joblib.load(f"{MODELS_DIR}/L03_{MY_NAME}_best.joblib"); model.predict(date_noi)
```

**De ce `cv_results_final.loc[tuned, coloana_noua] = value` funcționează:** pandas adaugă automat rândul `tuned` și coloana nouă, iar celelalte celule devin `NaN`. Nu mai trebuie să le inițializezi tu cu `np.nan`.

---

## TODO 5 — Observații test (celula 61) ⚠️

### Ce e greșit
- *„Scorul de test este puțin mai mare pentru că e o diferență în limita zgomotului”*: concluzia e bună, dar fără cifre e doar o afirmație. **De ce** e zgomot? Pentru că testul are doar 63 de clienți din clasa 1. Fiecare client din clasa 1 ghicit sau ratat mută recall-ul cu 1/63 = 1,6 puncte și acuratețea echilibrată cu ≈ 0,8 puncte.
- *„Am câștigat cu +0,115”*: cifra e corectă (L02: 0,709 → L03: 0,824 acuratețe echilibrată pe test), dar întrebarea este **din ce anume**. Răspunsul se vede în tabelele tale: aproape tot câștigul vine din `class_weight` (0,687 → 0,798 în CV); hiperparametrii aduc ≈ 0, iar preprocesarea mai bună (ordinal, Robust) foarte puțin.

### Răspuns corect (cifrele mele: test 0,845, F1 0,617, ROC AUC 0,917; înlocuiește-le cu ale tale)
```markdown
**Observații:**
- **Test față de CV:** test 0,845 față de CV 0,808 și nested CV 0,798 ± 0,038. Testul iese puțin mai mare, dar diferența față de nested CV (0,047) e cam o abatere standard, deci e în limita zgomotului. Testul are doar 63 de clienți din clasa 1: fiecare client ghicit sau ratat mută recall-ul cu 1,6 puncte și acuratețea echilibrată cu ≈ 0,8 puncte. Un test „ușor” sau „greu” cu 4–5 clienți explică diferența. Nu am modificat nimic după ce am văzut testul.
- **Față de primul model din L02** (regresie logistică fără reechilibrare, acuratețe echilibrată pe test 0,753): câștig de **+0,092**. Câștigul vine aproape în întregime din **reechilibrare** (`class_weight="balanced"`: de la 0,687 la 0,798 în CV). Recall-ul pe clasa 1 a urcat de la 0,52 la 0,86, cu prețul unei precizii mai mici (0,85 → 0,48). Căutarea (C, scaler, `add_indicator`) aduce doar +0,010 în CV, sub abaterea standard.
- **În continuare:** (1) atribute noi: vârsta (`2014 − Year_Birth`), total copii (`Kidhome + Teenhome`), număr de campanii acceptate anterior (suma `AcceptedCmp*`), cheltuiala medie pe cumpărătură; curba de învățare arată că modelul e limitat de bias, deci atributele ajută mai mult decât datele; (2) alegerea pragului de decizie (5.2); (3) gradient boosting cu tuning: are ROC AUC aproape egal, deci cu prag ales poate ajunge la nivelul regresiei logistice.
```

> Cum am obținut 0,753 pentru L02: e acuratețea echilibrată a regresiei logistice din tabelul L02 **după curățarea corectată**. Cu tabelul tău vechi era 0,709 (0,5 + 0,209), de unde și +0,115 al tău.

---

## TODO 5.2 — Pragul de decizie (opțional) — cod

Nu l-ai făcut. Merită: e un câștig pe F1 la cost zero. Pragul se alege **doar din train** (predicții *out-of-fold*), apoi se aplică o dată pe test:
```python
# OPTIONAL: pragul care maximizeaza F1, ales din predictii out-of-fold pe TRAIN
from sklearn.model_selection import cross_val_predict

proba_oof = cross_val_predict(best_model, X_train, y_train, cv=cv5, method="predict_proba")[:, 1]
thresholds = np.arange(0.05, 0.96, 0.05)
f1_by_t = [f1_score(y_train, (proba_oof >= t).astype(int)) for t in thresholds]
best_t = thresholds[int(np.argmax(f1_by_t))]
print(f"train (OOF) F1 at 0.50: {f1_score(y_train, (proba_oof >= 0.5).astype(int)):.3f}  |  "
      f"best threshold {best_t:.2f}: F1 = {max(f1_by_t):.3f}")

# aplicat pe test (pragul a fost ales FARA test)
y_pred_t = (y_proba_te >= best_t).astype(int)
print(f"TEST F1 at {best_t:.2f}: {f1_score(y_test, y_pred_t):.3f}  (at 0.50: {test_scores['F1']:.3f})")
print(classification_report(y_test, y_pred_t, digits=3))
```
La mine: pragul ales este 0,60 (peste 0,5, pentru că `class_weight="balanced"` a împins deja probabilitățile în sus). Pe train (OOF), F1 crește doar de la 0,566 la 0,570, dar pe test de la 0,617 la 0,684 (recall 0,86 → 0,83, precizie 0,48 → 0,58). **Atenție la interpretare:** câștigul real așteptat e cel din OOF (≈ 0); saltul de pe test e în mare parte noroc (63 de pozitive). Raportează-l așa, nu ca pe o îmbunătățire sigură.

---

## Celulele AI (celulele 25, 33, 42, 48, 55, 63) ⚠️

Toate au **„Rezultat:” gol**, iar „Promptul folosit” are 3–4 cuvinte (*„Compara strategiile”*, *„Sa realizeze cele doua grafice”*). Laboratorul cere promptul cu cei 4 pași (context, cerință, constrângeri, verificare). Exemplu pentru celula 33:
```markdown
- **Promptul folosit:** „Am un DataFrame `X_train` (1646 × 25) cu ținta binară `Response` (15% clasa 1) și o funcție `make_pipeline` care întoarce un Pipeline prep + model. Vreau să compar strategiile none, class_weight, oversampling și SMOTE pe același `StratifiedKFold(5, shuffle=True, random_state=SEED)`, cu un `imblearn` Pipeline (prep → resample → model), pe recall, precision, F1 și balanced accuracy. SMOTE doar în interiorul pipeline-ului. Explică fiecare linie.”
- **Codul primit:** dicționarul `strategies` și bucla cu `cross_validate`.
- **Rezultat:** tabelul de la 2.6; class_weight / oversampling / SMOTE la egalitate (≈ 0,80 bal. acc.), toate peste „none” (0,69).
- **Cum am verificat:** am verificat că pasul `resample` apare doar în `ImbPipeline`, nu înainte de `cross_validate`; am comparat cu exemplul Penguins (aceeași structură).
```
Dacă nu ai folosit AI la un pas, scrie asta sau șterge celula. Nu lăsa o celulă AI cu bucăți din codul tău lipite în ea.
