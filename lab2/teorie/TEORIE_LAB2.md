# Teorie — Laboratorul 2: încărcarea, structura, explorarea și curățarea datelor

> Adaptat după `lab2/TEORIE_LAB2.md` din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB) (acolo pe Telco Churn), refăcut pe **setul nostru: Customer Personality Analysis**. Toate cifrele sunt calculate pe `lab2/data/raw.csv` (`python lab2\todo3.py`).

Laboratorul 2 este **primul pas** din orice proiect de machine learning: înainte să construim un model, trebuie să **cunoaștem datele**. Ce coloane avem, ce tip au, ce e greșit sau lipsă în ele și ce legături există între ele.

Fiecare termen are:
- **Ce înseamnă** — definiția, în cuvinte simple;
- **La noi** — un exemplu din setul Customer Personality;
- **În cod** — unde apare în `todo1.py`, `todo2.py`, `todo3.py` sau în notebook.

La final (secțiunea 15) ai **codul laboratorului explicat bucată cu bucată**, un tabel-rezumat și întrebări de verificare. Explicația fiecărei coloane e în [00_COLOANE_SET_DATE.md](00_COLOANE_SET_DATE.md).

**Cuprins**
1. Vocabularul de bază
2. Tipurile de probleme: clasificare și regresie
3. Tipurile de coloane
4. Structura datelor: cardinalitate, constante, duplicate, tipuri greșite
5. Valori lipsă și coduri ascunse
6. Statistici descriptive: centrul
7. Statistici descriptive: împrăștierea
8. Forma distribuției: skew și kurtosis
9. Outlieri și regula IQR
10. Distribuția țintei și dezechilibrul claselor
11. Legătura dintre coloane: corelații, tabele de contingență
12. Data leakage
13. Curățarea datelor
14. Primele modele (I · C · T)
15. **Codul din Lab 2, explicat**
16. Termeni Python / pandas
17. Rezumat + întrebări de verificare

---

## 1. Vocabularul de bază

### Machine learning (învățare automată)
**Ce înseamnă:** în loc să scriem noi regulile („dacă clientul a acceptat campania 5, atunci…”), îi dăm calculatorului **exemple** și el găsește singur regulile. Un **model** este „formula” învățată din date: primește informații despre un client și dă un răspuns.

### Învățare supervizată
**Ce înseamnă:** exemplele vin **cu răspunsul corect**. Pentru fiecare client din tabel știm dacă a acceptat ultima ofertă (`Response`) și cât a cheltuit (`Mnt*`), deci modelul poate învăța din ele.

### Set de date (dataset)
**La noi:** **2240 de rânduri × 29 de coloane**, în `raw.csv`.

### Rând (observație, instanță)
**La noi:** un rând = **un client** al lanțului de magazine.

### Coloană (atribut, feature, variabilă)
**La noi:** `Income`, `Recency`, `MntWines`, `AcceptedCmp5` etc.

### Target (ținta, `y`)
**Ce înseamnă:** coloana pe care vrem să o **prezicem**. Restul coloanelor folosite pentru predicție = **atribute**, notate **X**.
**La noi:**
- `Response` (a acceptat clientul ultima ofertă? 0/1) — pentru echipa de marketing care alege cui trimite oferta;
- `MntTotal` (cât a cheltuit în 2 ani) — o coloană pe care **o construim noi** ca sumă a celor 6 coloane `Mnt*`.

**În cod:** `TARGET_CLS = "Response"` în `todo2.py`; `df["MntTotal"] = df[spend_cols].sum(axis=1)` în `todo3.py`.

### „Povestea” setului
**Ce înseamnă:** cine folosește modelul și pentru ce. Povestea decide ce e ținta, ce coloane avem voie să folosim și ce greșeală e mai gravă.
**La noi:** firma pregătește o campanie. Fiecare contact costă **3** (`Z_CostContact`), fiecare acceptare aduce **11** (`Z_Revenue`). E mai grav să **ratezi** un client care ar fi acceptat (pierzi 11) decât să contactezi degeaba unul care refuză (pierzi 3).

---

## 2. Tipurile de probleme

### Clasificare
**Ce înseamnă:** ținta este o **categorie**. Modelul răspunde „în ce grup intră?”.
**La noi:** `Response` (0 / 1) → **clasificare binară**.

### Regresie
**Ce înseamnă:** ținta este un **număr continuu**. Modelul răspunde „cât?”.
**La noi:** `MntTotal` (între 5 și 2525) → **regresie**.

> „Regresia **logistică**” este un model de **clasificare**, deși are „regresie” în nume.

### Clasa pozitivă
**Ce înseamnă:** clasa care ne interesează, notată cu 1. Nu „bună”, ci „cea căutată”.
**La noi:** `Response = 1` (clientul acceptă oferta).

---

## 3. Tipurile de coloane

### Coloană numerică
**Ce înseamnă:** **numere care au sens ca mărimi** (putem face media, putem spune „de două ori mai mult”).
**La noi:** `Income`, `Recency`, `Mnt*`, `Num*`, `Year_Birth`.
- **discretă:** numere întregi, numărabile (`Kidhome`: 0, 1, 2; `NumWebPurchases`);
- **continuă:** orice valoare (`Income`).

### Coloană categorială
**Ce înseamnă:** conține **etichete**, nu mărimi.
**La noi:** `Education`, `Marital_Status`.
**În cod:** `df[rest].select_dtypes(exclude="number")` în `split_columns` (`todo2.py`).
- **Nominală:** fără ordine — `Marital_Status` (Married, Single… niciuna nu e „mai mare”);
- **Ordinală:** cu ordine naturală — `Education`: Basic < Graduation < Master (2n Cycle) < PhD. Contează la codificare (Lab 3).

### Coloană binară
**Ce înseamnă:** exact 2 valori.
**La noi:** `AcceptedCmp1–5`, `Complain`, `Response` (toate 0/1).
**În cod:** `binary = nunique[nunique == 2].index.tolist()` în `todo2.py`.
Spre deosebire de `SeniorCitizen` la Telco, coloanele noastre 0/1 **pot rămâne numerice**: one-hot pe o coloană 0/1 nu aduce nimic nou.

### Un număr care e o numărătoare, nu o categorie
`Kidhome` și `Teenhome` (0, 1, 2) **rămân numerice**: 2 copii chiar înseamnă de două ori mai mulți decât 1. Sunt numerătoare cu distanțe reale.

### Identificator (ID)
**Ce înseamnă:** o valoare diferită pe fiecare rând, doar ca să deosebim rândurile.
**La noi:** `ID`: **2240 de valori unice din 2240** de rânduri.
**De ce contează:** un ID nu descrie clientul; modelul l-ar putea „memora”. Se elimină.
**În cod:** `id_like = card.index[card["n_unique"] == n]` în `todo2.py`.

### Coloană de dată
**La noi:** `Dt_Customer` (data înscrierii). Nu e nici numerică, nici categorială: o transformăm într-un număr (`Customer_Days` = câte zile e client).

---

## 4. Structura datelor

### Tipul de date (dtype)
- `int64`: întreg; `float64`: cu zecimale; `object`: text.

**La noi:** 25 de coloane `int64`, 1 `float64` (`Income`), 3 `object` (`Education`, `Marital_Status`, `Dt_Customer`).
**În cod:** `df.dtypes.astype(str).value_counts()` în `todo2.py`.

### Tip greșit
**La noi:**
- `Dt_Customer` e **text** (`"04-09-2012"`), dar e o **dată** → `pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")`;
- `Income` e `float64` doar din cauza celor 24 de `NaN` (în pandas, `NaN` e un float). Tipul e corect.

### Cardinalitate
**Ce înseamnă:** **numărul de valori diferite** dintr-o coloană. **În cod:** `df.nunique()`.

| Cardinalitate | Coloane | Ce înseamnă |
|---|---|---|
| 1 | `Z_CostContact` (=3), `Z_Revenue` (=11) | **constante**, inutile |
| 2 | `AcceptedCmp1–5`, `Complain`, `Response` | **binare** |
| 3 | `Kidhome`, `Teenhome` | numărători mici |
| 5 / 8 | `Education` / `Marital_Status` | categoriale cu cardinalitate mică |
| 14–16 | `Num*` | numerice discrete |
| 100 | `Recency` | 0–99 zile |
| 663 | `Dt_Customer` | date (text) |
| 1974 | `Income` | numerică continuă |
| 2240 (= nr. rânduri) | `ID` | **identificator** |

### Coloană constantă
**La noi:** `Z_CostContact` = 3 și `Z_Revenue` = 11 pe **toate** rândurile → nu pot deosebi un client de altul → se elimină. (Sunt „constantele” campaniei, utile doar pentru calculul profitului.)
**Aproape constantă:** `Complain`: doar 21 de clienți (0,9%) s-au plâns.
**În cod:** `constant = nunique[nunique == 1].index.tolist()`.

### Duplicate
**La noi:**
- `df.duplicated().sum()` = **0**: `ID` e diferit peste tot;
- **fără `ID`**: **182** de rânduri identice pe toate cele 28 de coloane rămase. Spre deosebire de Telco (unde erau 22 de profiluri simple), aici rândurile au 28 de valori, inclusiv sume exacte cheltuite pe 6 categorii — e practic imposibil ca doi clienți diferiți să coincidă întâmplător. Sunt **același client introdus de două ori** → se elimină. Altfel, același client poate ajunge și în train, și în test, iar scorul iese prea optimist.

**În cod:** `df.drop(columns=ID_COL).duplicated().sum()` în `todo2.py`.

### Redundanță
**La noi:** `MntTotal` = exact suma celor 6 `Mnt*`. La clasificare, dacă `Mnt*` sunt deja în X, `MntTotal` nu aduce nimic nou → o scoatem.

---

## 5. Valori lipsă și coduri ascunse

### Valoare lipsă (NaN)
**În cod:** `df.isna().sum()`, `100 * df.isna().mean()`.
**La noi:** doar `Income`: **24 de valori lipsă (1,07%)**, adică 24 din 71.680 de celule.
Nu e o lipsă „structurală” cu explicație (ca la Telco, unde lipseau la clienții noi): pur și simplu nu s-a înregistrat venitul → completăm cu **mediana** (Lab 3), în pipeline.

### Cod ascuns (valoare-santinelă)
**Ce înseamnă:** o valoare care **arată normal**, dar înseamnă „lipsă” sau „eroare”.
**La noi:**
- `Income = 666666`: cifre repetate, de 4 ori mai mare decât următorul venit (162.397) → cod de umplere → `NaN`;
- `Year_Birth` = **1893, 1899, 1900** → vârste de 114–121 de ani, imposibile → `NaN`;
- `Marital_Status` = **Absurd**, **YOLO** (2 + 2) → răspunsuri fără sens → „Other”;
- `Marital_Status` = **Alone** (3) → sinonim pentru Single;
- `Education` = **2n Cycle** (203) → același nivel ca Master, scris altfel.

**În cod:** dicționarul `hidden` din `todo2.py`, secțiunea 5c.

---

## 6. Statistici descriptive: centrul unei coloane

**În cod:** `df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])` în `todo3.py`.

### Media
**La noi:** `MntTotal` are media **606**.
**Atenție:** sensibilă la valorile extreme. `Income` are media 52.247, „trasă” de valoarea 666666.

### Mediana
**La noi:** `MntTotal` are mediana **396**.
**Media vs mediana:** 606 ≫ 396 → câțiva clienți cheltuie foarte mult și „trag” media în sus → asimetrie spre dreapta. La `MntWines`: 304 vs 174; la `MntMeatProducts`: 167 vs 67.

### Modul
**La noi:** `Education` → Graduation (1127 de clienți); `Marital_Status` → Married (864).
**În cod:** pentru coloanele text, `describe()` dă `top` (= modul) și `freq`.

---

## 7. Statistici descriptive: împrăștierea

### Min, max, amplitudine
**La noi:** `Recency` 0–99 zile; `MntTotal` 5–2525; `Income` 1730–666666 (max = eroarea).

### Varianța și abaterea standard (std)
**La noi:** `Recency`: media 49, std 29. `MntTotal`: media 606, std **602** — std cât media → foarte împrăștiată.

### Coeficientul de variație (CV = std / medie)
**La noi:** `MntTotal` CV **0,99**; `Mnt*` între 1,1 și 1,5 (foarte împrăștiate); `Age` 0,27; `NumWebVisitsMonth` 0,46 (clienții seamănă între ei).
**În cod:** `"CV": s.std() / s.mean()` în `numeric_summary`.

### Percentile, cuartile, IQR
**La noi:** `MntTotal`: Q1 = 69, Q2 = 396, Q3 = 1046 → IQR = **977**. `Income`: Q1 = 35.303, Q3 = 68.522 → IQR = 33.219.
**În cod:** `q1, q3 = s.quantile([0.25, 0.75])`; `iqr = q3 - q1`.

---

## 8. Forma distribuției

### Asimetria (skew)
- skew ≈ 0 → simetrică; > 0 → coadă spre dreapta (media > mediana); < 0 → coadă spre stânga.
- |skew| < 0,5 aproape simetrică; 0,5–1 moderat; > 1 puternic.

**La noi:**
| Coloană | skew | Interpretare |
|---|---|---|
| `Income` | **6,76** | extremă, **doar din cauza lui 666666**; fără el, aproape simetrică |
| `MntFruits`, `MntSweetProducts`, `MntMeatProducts` | 2,1 | puternic spre dreapta |
| `NumDealsPurchases` | 2,42 | puternic spre dreapta |
| `MntWines` | 1,18 | puternic |
| `MntTotal` | **0,86** | moderat spre dreapta |
| `Recency`, `Customer_Days` | ≈ 0 | simetrice (aproape uniforme) |

**De ce contează:** pentru o coloană asimetrică, mediana descrie mai bine clientul tipic; la imputare preferăm mediana; `np.log1p` poate face o coloană (sau ținta `MntTotal`, Lab 5) mai simetrică.

### Kurtosis
**La noi:** `Income` **159,6** (o valoare extremă imensă), `NumDealsPurchases` 8,9, `NumCatalogPurchases` 8,1 (cozi grele); `Recency` −1,20 și `Customer_Days` −1,19 (distribuții „turtite”, aproape uniforme).

---

## 9. Outlieri

### Regula IQR (Tukey)
Candidat la outlier dacă e sub **Q1 − 1,5·IQR** sau peste **Q3 + 1,5·IQR**.
**În cod:** `outside = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()`.
**Important:** doar un **candidat**; se verifică, nu se șterge automat.

**La noi:**
| Coloană | Candidați IQR | Verdict |
|---|---|---|
| `Income` | 8 | 1 eroare (666666); restul (~150.000–162.000) sunt venituri mari, dar reale |
| `Age` | 3 | exact cei 3 născuți în 1893–1900 → **erori** |
| `MntFruits`, `MntSweetProducts`, `MntFishProducts`, `MntGoldProds` | 200–250 fiecare | **cazuri reale**: distribuția are multe zerouri și Q3 mic, deci orice cumpărător mare iese „outlier”. Nu se șterg |
| `NumDealsPurchases` | 86 | reali (vânători de reduceri) |
| `MntTotal` | 3 | reali |

### Boxplot
Arată mediana, Q1/Q3 (cutia), mustățile (1,5·IQR) și outlierii (punctele).

---

## 10. Distribuția țintei și dezechilibrul

**În cod:** `y.value_counts(normalize=True)`.
**La noi:** **85,1% nu acceptă** (1906), **14,9% acceptă** (334). Raport de dezechilibru = 1906 / 334 = **5,71** (Telco: 2,77 — al nostru e de două ori mai dezechilibrat).

**De ce contează:** un model „leneș” care spune mereu 0 are acuratețea **85,1%** fără să găsească **niciun** client interesat. Acuratețea înșală → folosim **precizie**, **recall**, **F1**, **acuratețe echilibrată**, **ROC-AUC / PR-AUC** (Lab 3–4).

### Statistici pe clasă
**În cod:** `df.groupby(TARGET)[cols].agg(["mean", "median"])`.

| Coloană (mediana) | Response = 0 | Response = 1 | Ce spune |
|---|---|---|---|
| `Income` | 50.150 | **64.090** | cei care acceptă câștigă mai mult |
| `Recency` | 52 | **30** zile | au cumpărat mai recent |
| `Customer_Days` | 331 | **491** | sunt clienți mai vechi |
| `MntWines` | 152 | **448** | cumpără mult mai mult vin |
| `MntTotal` | 314 | **1058** | cheltuie de 3 ori mai mult |
| `NumCatalogPurchases` | 1 | **4** | cumpără din catalog |
| `Age` | 44 | 43 | **nicio diferență** → vârsta nu ajută |

### Rata de răspuns pe categorie
| Coloană | Rata de răspuns pe valori | Spread |
|---|---|---|
| `AcceptedCmp2` | 0 → 14,2%, 1 → **66,7%** | **52,5 pp** |
| `AcceptedCmp5` | 11,7% / **56,4%** | 44,7 pp |
| `AcceptedCmp1` | 12,2% / **54,9%** | 42,7 pp |
| `Education` | Basic 3,7% … PhD **20,8%** | 17,1 pp |
| `Kidhome` | 0 copii 17,2%, 2 copii 4,2% | 13,0 pp |
| `Marital_Status` | Married 11,3%, Single 22,1%, Widow 24,7% (Absurd/YOLO 50% din 2 clienți — zgomot) | — |
| `Complain` | 14,9% / 14,3% | **0,6 pp** → inutilă |

**Concluzie:** cine a acceptat o campanie **anterioară** acceptă de 4–5 ori mai des și pe cea nouă.

---

## 11. Legătura dintre coloane

### Corelația (−1 … +1)
|r| < 0,1 → deloc; 0,1–0,3 → slabă; 0,3–0,5 → moderată; > 0,5 → puternică.

### Pearson vs Spearman
- **Pearson:** legătură **liniară** (puncte pe o dreaptă), sensibilă la outlieri;
- **Spearman:** legătură **monotonă**, pe ranguri, robustă la outlieri. La noi e mai potrivită: coloanele `Mnt*` sunt puternic asimetrice, iar `Income` are 666666.

**În cod:** `corr_df.corr(method="pearson")` / `method="spearman"`.

**La noi — cele mai corelate perechi de atribute (Pearson):**
- `MntMeatProducts` – `NumCatalogPurchases` **0,72**;
- `MntWines` – `NumStorePurchases` / `NumCatalogPurchases` 0,64;
- `Income` – `NumCatalogPurchases` 0,59; `Income` – `MntWines` 0,58.
→ cine are venit mare cumpără mult, mai ales din catalog, mai ales vin și carne.

**Corelația cu `Response` (Spearman):** `AcceptedCmp5` **0,33**, `AcceptedCmp1` 0,29, `AcceptedCmp3` 0,25, `MntTotal` 0,24, `NumCatalogPurchases` 0,23, `Recency` **−0,20**, `Teenhome` −0,16; `Age` −0,02 și `Complain` 0,00 → practic nimic.

**Corelația cu `MntTotal` (Spearman):** `NumCatalogPurchases` **0,89**, `Income` **0,85**, `NumStorePurchases` 0,80, `NumWebPurchases` 0,73, `Kidhome` **−0,62**, `NumWebVisitsMonth` −0,48.

### Corelație ≠ cauzalitate
`NumWebVisitsMonth` e corelată negativ cu cheltuiala. Nu înseamnă că vizitele pe site îi fac pe clienți să cheltuie mai puțin: cei cu venit mic și copii intră des să caute oferte.

### Tabel de contingență (crosstab)
**În cod:** `pd.crosstab(df["AcceptedCmp5"], df[TARGET], margins=True)`.
**La noi:**
| AcceptedCmp5 | Response = 0 | Response = 1 | Total |
|---|---|---|---|
| 0 | 1835 | 242 | 2077 |
| 1 | 71 | **92** | 163 |
| Total | 1906 | 334 | 2240 |

Din cei 163 care au acceptat campania 5, **92 (56%)** acceptă și oferta nouă.

---

## 12. Data leakage (scurgerea de informație)

**Ce înseamnă:** modelul primește o informație pe care **n-ar avea-o în realitate** în momentul predicției, de obicei pentru că conține deja răspunsul. Scorul iese foarte bun, dar e **fals**.

**Semnale:** corelație atribut–țintă aproape de ±1; o categorie cu ținta 0% / 100%.

**La noi:**
- **clasificare (`Response`):** maximum |0,33|, nicio categorie cu 0% / 100% → fără leakage evident. `AcceptedCmp1–5` **nu** sunt leakage: sunt campanii **anterioare**, le știm înainte de cea nouă;
- **regresie (`MntTotal`):**
  - `Mnt*` → suma lor **este** ținta (corelație **1,0**) → se scot **toate**;
  - `NumWeb/Catalog/Store/DealsPurchases` → numărate în **aceeași perioadă de 2 ani** ca `MntTotal` (corelație 0,91). Le știm abia **după** ce clientul a cheltuit → le scoatem și pe ele (R² scade de la ~0,85 la ~0,76, dar scorul devine cinstit);
  - `Response` → rezultatul campaniei, nu îl știm pentru un client nou → se scoate.

**În cod:** secțiunea 5 din `todo3.py`.

---

## 13. Curățarea datelor

**Regula de aur:** curățăm pe o **copie** (`df_clean = df.copy()`), iar deciziile le scriem **înainte** de cod.

| # | Decizie | Motiv |
|---|---|---|
| 1 | scoatem `ID` | identificator |
| 2 | scoatem `Z_CostContact`, `Z_Revenue` | constante |
| 3 | `drop_duplicates()` după ce scoatem ID-ul | 182 de clienți introduși de două ori |
| 4 | `Income = 666666` → `NaN` | cod de umplere; rândul rămâne (ținta e validă) |
| 5 | `Year_Birth < 1920` → `NaN` | ani imposibili |
| 6 | `Alone` → `Single`; `Absurd`, `YOLO` → `Other` | sinonim / fără sens |
| 7 | `2n Cycle` → `Master` | același nivel (Bologna) |
| 8 | `Dt_Customer` → `Customer_Days` (față de o dată **fixă**, 31.12.2014) | dată → număr |
| 9 | `MntTotal` = suma `Mnt*` | ținta de regresie |
| 10 | **nu** completăm `NaN` acum | mediana se învață doar pe train (altfel leakage) |

**Rezultat:** `(2240, 29)` → **`(2058, 27)`**; rămân lipsă `Income` 25 (24 + 666666) și `Year_Birth` 3. Se salvează în `clean.csv`.

---

## 14. Primele modele (I · C · T)

- **I – Isolate:** `train_test_split(test_size=0.2, stratify=y, random_state=SEED)`. `stratify=y` păstrează 14,9% clasa 1 și în train, și în test. La regresie nu se stratifică (ținta e numerică).
- **C – Construct:** `ColumnTransformer` + `Pipeline`: numerice → mediană + `StandardScaler`; categoriale → cea mai frecventă + `OneHotEncoder`.
- **T – Test:** comparăm cu un **baseline** (`DummyClassifier` / `DummyRegressor`).

**Rezultate (test, după curățare):**
| Clasificare | Acuratețe | Acuratețe echilibrată | F1 | ROC-AUC |
|---|---|---|---|---|
| baseline (mereu 0) | 0,847 | 0,500 | 0,000 | 0,500 |
| **regresie logistică** | 0,913 | **0,753** | **0,647** | **0,912** |
| kNN (k=5) | — | 0,646 | 0,435 | 0,798 |
| arbore (max_depth=3) | — | 0,621 | 0,386 | 0,769 |

Regresie (`MntTotal`, fără `Mnt*` și `Num*Purchases`): baseline MAE ≈ 477, R² ≈ 0 → regresie liniară MAE ≈ 210, **R² ≈ 0,76**.

**Ce se vede:** acuratețea câștigă doar +0,066 față de baseline, dar F1 câștigă +0,647 → la clase dezechilibrate acuratețea ascunde aproape tot. Recall-ul clasei 1 e doar ~0,52 → în Lab 3 reechilibrăm clasele.

---

## 15. Codul din Lab 2, explicat

### 15.1 `todo1.py` — încărcarea datelor

```python
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LAB_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
```
- `__file__` = calea scriptului; `abspath` o face completă; `dirname` păstrează doar folderul → `LAB_DIR` = folderul `lab2`, oriunde ai rula scriptul.
- `os.makedirs(..., exist_ok=True)` creează folderul `data` dacă nu există, fără eroare dacă există.

```python
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
```
Pandas ascunde coloanele cu „…” când sunt multe; așa le vezi pe toate 29.

```python
def load_data():
    raw_csv = os.path.join(DATA_DIR, "raw.csv")
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
    else:
        import kagglehub
        p = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
        f = glob.glob(p + "/**/marketing_campaign.csv", recursive=True)[0]
        df = pd.read_csv(f, sep="\t")
        df.to_csv(raw_csv, index=False)
    return df
```
- **Cache:** dacă `raw.csv` există, îl citește de pe disc (rapid, fără internet). Altfel îl descarcă **o singură dată**.
- `kagglehub.dataset_download` întoarce un **folder**, nu un fișier → `glob` cu `/**/` și `recursive=True` caută fișierul și în subfoldere; `[0]` ia primul rezultat.
- `sep="\t"`: fișierul original e separat cu **TAB**, nu virgulă. Fără asta, toată linia ar intra într-o singură coloană.
- `to_csv(index=False)`: salvează copia locală fără coloana de index a lui pandas.
- O **funcție** (`load_data`) ca celelalte TODO-uri să o refolosească: `from todo1 import load_data`.

```python
if __name__ == "__main__":
    df = load_data()
    assert df.shape == (2240, 29), f"Forma neasteptata: {df.shape}"
```
- Blocul rulează doar când pornești direct `todo1.py`, nu când e importat.
- `assert` oprește programul cu mesaj dacă forma nu e cea din fișa setului (2240 × 29) — prinde un fișier greșit sau citit greșit.

### 15.2 `todo2.py` — structura setului

```python
ID_COL = "ID"
TARGET_CLS = "Response"
DATE_COLS = ["Dt_Customer"]
```
Constante definite **o singură dată**, sus; restul codului le folosește prin nume.

```python
def split_columns(df):
    spend = [c for c in df.columns if c.startswith("Mnt")]
    special = [ID_COL, TARGET_CLS] + DATE_COLS
    rest = [c for c in df.columns if c not in special]
    numeric = df[rest].select_dtypes(include="number").columns.tolist()
    categorical = df[rest].select_dtypes(exclude="number").columns.tolist()
    nunique = df[numeric].nunique()
    constant = nunique[nunique == 1].index.tolist()
    binary = nunique[nunique == 2].index.tolist()
    ...
```
- **List comprehension** `[c for c in df.columns if c.startswith("Mnt")]` = „toate coloanele al căror nume începe cu Mnt” → cele 6 coloane de cheltuieli.
- Întâi separă coloanele **pe roluri** (ID, țintă, dată), apoi restul **pe tipuri** cu `select_dtypes`.
- Cardinalitatea (`nunique`) separă numericele în constante (1 valoare), binare (2) și restul.
- Întoarce un **dicționar** `{grup: [coloane]}`, refolosit de TODO-urile următoare.

```python
base = groups["id"] + groups["target_cls"] + groups["date"] + groups["categorical"] + groups["numeric"]
assert sorted(base) == sorted(df.columns)
```
Verificare: grupurile de bază acoperă **exact** toate cele 29 de coloane, fiecare o singură dată. Dacă ai uitat o coloană sau ai pus-o de două ori, `assert` te anunță.

```python
card = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_unique": df.nunique(),
    "unique_%": (df.nunique() / n * 100).round(2),
    "missing": df.isna().sum(),
    "top_value": df.mode().iloc[0],
    "top_%": (df.apply(lambda s: s.value_counts(normalize=True).iloc[0]) * 100).round(1),
}).sort_values("n_unique")
```
Un tabel cu un rând pe coloană: tipul, cardinalitatea, cât % din rânduri sunt valori unice, câte lipsesc, valoarea cea mai frecventă și ce procent are. `df.mode().iloc[0]` = modul fiecărei coloane; `lambda s: ...` = o funcție mică aplicată pe fiecare coloană. `top_%` mare (de ex. 99%) = coloană aproape constantă.

```python
dup_rows = df.duplicated().sum()                               # 0
df.drop(columns=ID_COL).duplicated().sum()                     # 182
```
`duplicated()` marchează cu `True` rândurile care repetă un rând anterior; `.sum()` le numără (True = 1). Scoțând `ID`-ul apar duplicatele „ascunse”.

```python
hidden = {
    "Marital_Status = Absurd / YOLO": df["Marital_Status"].isin(["Absurd", "YOLO"]),
    "Income = 666666 ...": df["Income"].eq(666666),
    "Year_Birth < 1920 ...": df["Year_Birth"].lt(1920),
    ...
}
```
Fiecare valoare e o **mască booleană** (o coloană de True/False). `isin` = „e în listă”, `eq` = „egal cu”, `lt` = „mai mic decât”. `df.loc[mask, ID_COL]` arată ID-urile rândurilor suspecte.

```python
purchases = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
(purchases == 0) & (spend_total > 0)
```
`sum(axis=1)` adună **pe rând** (pe orizontală). Verifică inconsistențe logice: un client cu 0 cumpărări, dar cu bani cheltuiți. `&` = „și” între două măști.

### 15.3 `todo3.py` — statistici, corelații, leakage

```python
df["MntTotal"] = df[spend_cols].sum(axis=1)
df["Age"] = 2014 - df["Year_Birth"]
dt = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["Customer_Days"] = (dt.max() - dt).dt.days
```
- Construiește ținta de regresie și două atribute mai ușor de interpretat: vârsta și vechimea.
- `format="%d-%m-%Y"` = zi-lună-an, explicit (fără să ghicească). Diferența a două date e o durată; `.dt.days` o transformă în număr de zile.

```python
def numeric_summary(data):
    for c in data.select_dtypes(include="number").columns:
        s = data[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows[c] = {"missing %": ..., "mean": ..., "median": ..., "std": ..., "CV": ...,
                   "skew": s.skew(), "kurtosis": s.kurt()}
    return pd.DataFrame(rows).T.round(2)
```
Pentru fiecare coloană numerică calculează toate statisticile din secțiunile 6–9 într-un singur tabel. `|` = „sau” între măști; `.T` = transpune tabelul (un rând pe coloană).

```python
counts = y.value_counts()
counts.max() / counts.min()                                     # 5.71
df.groupby(TARGET)[cols].agg(["mean", "median"])
```
Raportul de dezechilibru și statisticile **separat** pentru `Response = 0` și `Response = 1`.

```python
for c in cat_cols:
    rate = (100 * y.groupby(df[c]).mean()).round(1)
    spread[c] = rate.max() - rate.min()
```
Media lui `y` (0/1) într-un grup = **procentul de 1** din grup = rata de răspuns. `spread` = cât de diferite sunt ratele între categorii; spread mare = atribut util.

```python
pearson = corr_df.corr(method="pearson")
p_attr.where(np.triu(np.ones(p_attr.shape, dtype=bool), k=1)).stack()
```
Matricea de corelații e simetrică (corr(A,B) = corr(B,A)). `np.triu(..., k=1)` păstrează doar triunghiul de **deasupra** diagonalei → fiecare pereche apare o singură dată, fără perechea „coloană cu ea însăși” (= 1). `stack()` transformă matricea într-o listă de perechi.

```python
pure = [(c, v, ...) for c in cat_cols for v, r in y.groupby(df[c]).mean().items() if r in (0, 1)]
df[TARGET_REG].corr(df[spend_cols].sum(axis=1))                 # 1.0 -> leakage
```
Caută categorii în care **toți** sau **niciunul** dintre clienți au răspuns (semnal de leakage) și confirmă că `MntTotal` e exact suma `Mnt*`.

### 15.4 Curățarea (notebook, TODO 5)

```python
df_clean = df.copy()
df_clean = df_clean.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])
df_clean = df_clean.drop_duplicates()
df_clean.loc[df_clean["Income"] == 666666, "Income"] = np.nan
df_clean.loc[df_clean["Year_Birth"] < 1920, "Year_Birth"] = np.nan
df_clean["Marital_Status"] = df_clean["Marital_Status"].replace({"Alone": "Single", "Absurd": "Other", "YOLO": "Other"})
df_clean["Education"] = df_clean["Education"].replace({"2n Cycle": "Master"})
signup = pd.to_datetime(df_clean["Dt_Customer"], format="%d-%m-%Y")
df_clean["Customer_Days"] = (pd.Timestamp("2014-12-31") - signup).dt.days
df_clean = df_clean.drop(columns=["Dt_Customer"])
df_clean["MntTotal"] = df_clean[MNT_COLS].sum(axis=1)
df_clean.to_csv(f"{MY_DIR}/clean.csv", index=False)
```
- `copy()` → datele brute rămân neatinse (poți compara înainte/după).
- `drop_duplicates()` **după** `drop(ID)` → altfel ID-ul diferit ar ascunde duplicatele.
- `df.loc[mască, coloană] = np.nan` = „pe rândurile unde masca e True, pune NaN în coloana asta”. Rândul **rămâne** (ținta și celelalte 25 de coloane sunt valide).
- `replace({vechi: nou})` înlocuiește valorile categoriale.
- Data **fixă** de referință (31.12.2014) → `Customer_Days` are același sens și pentru un client nou.
- Lipsurile **nu** se completează aici: completarea învață o mediană → se face în pipeline, doar pe train.

### 15.5 Primele modele (notebook, TODO 6)

```python
X = df_clean.drop(columns=[TARGET_C, TARGET_R])
y = df_clean[TARGET_C]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
```
- X = tot în afară de cele două ținte. `MntTotal` iese și din X-ul de clasificare (e suma `Mnt*`, care rămân).
- `test_size=0.2` → 20% test; `stratify=y` → aceeași proporție de 1 în train și test; `random_state=SEED` → aceeași împărțire la fiecare rulare.

```python
def make_preprocess(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
```
- `ColumnTransformer` = aplică **alte** transformări pe **alte** coloane și lipește rezultatele.
- Numerice: completează lipsurile cu mediana (robustă la 666666), apoi scalează (media 0, std 1).
- Categoriale: completează cu valoarea cea mai frecventă, apoi one-hot (o coloană 0/1 pe categorie). `handle_unknown="ignore"` → o categorie nouă în test nu dă eroare.

```python
for name, model in models_c.items():
    pipe = Pipeline([("prep", preprocess_c), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), ...,
                 "ROC AUC": roc_auc_score(y_test, proba)})
results_c = pd.DataFrame(rows).set_index("model").round(3)
```
- Fiecare model e pus în **același** pipeline → toate primesc exact aceeași preprocesare → comparație corectă.
- `fit` învață preprocesarea **și** modelul doar din train. `predict` dă clasa (0/1); `predict_proba(...)[:, 1]` dă probabilitatea clasei 1 (pentru ROC-AUC, care are nevoie de scoruri, nu de clase).
- `zero_division=0` la precizie: baseline-ul nu prezice niciun 1 → precizia ar fi 0/0.

```python
LEAKY_R = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
Xr = df_clean.drop(columns=[TARGET_R] + MNT_COLS + LEAKY_R)
```
La regresie X-ul e **altul**: scoatem ținta, componentele ei (`Mnt*`) și cumpărările din aceeași perioadă (leakage, secțiunea 12).

---

## 16. Termeni de Python / pandas

| Termen | Ce face |
|---|---|
| `pd.read_csv(f, sep="\t")` | citește un fișier separat cu TAB |
| DataFrame / Series | tabel / o singură coloană |
| `df.shape` | `(2240, 29)` |
| `df.info()`, `df.dtypes` | tipurile coloanelor |
| `df.describe()` | statisticile descriptive |
| `df.nunique()` | cardinalitatea |
| `df.isna()` | True unde lipsește o valoare |
| `df.duplicated()` | True pentru rândurile repetate |
| `value_counts(normalize=True)` | frecvențe în proporții |
| `groupby(col)` | calcule pe grupuri |
| `select_dtypes(include="number")` | doar coloanele numerice |
| `pd.to_datetime(s, format=...)` | text → dată |
| `s.isin([...])`, `s.eq(x)`, `s.lt(x)` | măști: „în listă”, „egal”, „mai mic” |
| `df.loc[mask, col] = val` | modifică doar rândurile din mască |
| `sum(axis=1)` | sumă pe rând |
| `pd.crosstab(a, b, margins=True)` | tabel de contingență cu totaluri |
| `if __name__ == "__main__":` | rulează doar la pornirea directă a fișierului |

---

## 17. Rezumat

| Termen | Pe scurt | La noi |
|---|---|---|
| target | ce prezicem | `Response` / `MntTotal` |
| clasificare / regresie | categorie / număr | acceptă oferta? / cât cheltuie? |
| numerică | mărimi | `Income`, `Recency`, `Mnt*`, `Num*` |
| categorială nominală / ordinală | fără / cu ordine | `Marital_Status` / `Education` |
| binară | 2 valori | `AcceptedCmp1–5`, `Complain` |
| identificator | unic pe rând | `ID` → eliminat |
| constantă | o valoare | `Z_CostContact`, `Z_Revenue` → eliminate |
| duplicate | rânduri identice | 0 cu ID, **182 fără ID** → eliminate |
| valoare lipsă | celulă goală | 24 în `Income` |
| cod ascuns | „lipsă” deghizată | 666666, 1893–1900, Absurd/YOLO |
| media / mediana | sensibilă / robustă | 606 / 396 la `MntTotal` |
| skew | asimetrie | `Income` 6,76 (outlier), `Mnt*` 1,2–2,1 |
| outlier IQR | în afara Q1−1,5·IQR … Q3+1,5·IQR | verificat: 4 erori, restul reali |
| dezechilibru | o clasă rară | 14,9% da, raport 5,71 |
| Pearson / Spearman | liniar / monoton | `MntTotal`–`Income` 0,85 |
| leakage | informația conține răspunsul | `Mnt*`, `Num*Purchases` la regresie |

### Întrebări de verificare

1. De ce scoatem `Z_CostContact` și `Z_Revenue`?
2. Am găsit 0 duplicate cu `ID` și 182 fără `ID`. Ce facem și de ce?
3. Media lui `MntTotal` e 606, mediana 396. Ce spune asta?
4. Un model care spune mereu 0 are acuratețea 85,1%. E bun?
5. De ce `Income = 666666` devine `NaN` în loc să ștergem rândul?
6. De ce `AcceptedCmp5` e permisă la clasificare, dar `MntWines` e interzisă la regresie?
7. De ce nu completăm lipsurile din `Income` în timpul curățării?

**Răspunsuri:**
1. Sunt constante (3 și 11 peste tot): nu pot deosebi clienții.
2. Le eliminăm: 28 de valori identice, inclusiv sume exacte, înseamnă același client introdus de două ori. Altfel poate ajunge și în train, și în test → scor optimist.
3. Distribuția e asimetrică spre dreapta: puțini clienți cheltuie foarte mult și trag media în sus.
4. Nu: recall 0, nu găsește niciun client interesat. La clase dezechilibrate acuratețea înșală.
5. E un cod de umplere, nu un venit; restul rândului (ținta, cheltuielile) e valid. Pierdem doar venitul, pe care îl completăm din train.
6. `AcceptedCmp5` e o campanie anterioară (o știm înainte). `MntWines` e o componentă a țintei `MntTotal` → ar fi ca și cum i-am da răspunsul.
7. Mediana se învață din date; dacă o calculăm pe tot setul, testul „scurge” informație în antrenare. Se face în pipeline, doar pe train.
