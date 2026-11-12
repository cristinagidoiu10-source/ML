# Context laboratoare ML

Fișier de context: tot ce primesc de la laboratoare (enunțuri, TODO-uri, exemple, decizii) se adaugă aici,
ca să pot răspunde consecvent la întrebările viitoare.

## Convenții generale (din notebook-uri)

- Mediu: Google Colab + Google Drive. Variabile definite în celulele de setup:
  `DATA_DIR`, `MY_DIR`, `FIGURES_DIR`, `RESULTS_DIR`, `MY_NAME`, `SEED`.
- Fiecare cerință vine cu un **EXAMPLE (Penguins)** și un **TODO** pe setul propriu. Soluția trebuie să
  **păstreze logica, ordinea și stilul exemplului** (comentarii în română fără diacritice, aceleași nume de pași).
- Salvări cerute:
  - figuri: `f"{FIGURES_DIR}/L02_{MY_NAME}_explore.png"` (salvare ÎNAINTE de `plt.show()`)
  - date curățate: `f"{MY_DIR}/clean.csv"`
  - rezultate: `f"{RESULTS_DIR}/L02_{MY_NAME}_first_models.csv"`
- Capcană Jupyter: doar ultima expresie din celulă se afișează → folosește `display(...)` pentru tabele la mijlocul celulei.
- `pd.set_option("display.max_columns", None)` ca să nu fie ascunse coloanele.

## Setul de date: Customer Personality Analysis

- Kaggle: `imakash3011/customer-personality-analysis`; `kagglehub.dataset_download` returnează un **director**.
  Fișierul: `marketing_campaign.csv`, separator **tab** (`sep="\t"`).
- Cache local: `f"{DATA_DIR}/customer/customer.csv"`; variabila de lucru: `customer` → `df = customer.copy()`.
- 2240 rânduri, 29 coloane:
  `ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer, Recency,
  MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds,
  NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth,
  AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain,
  Z_CostContact, Z_Revenue, Response`

### Ținte
- **Clasificare:** `Response` (0/1, a acceptat ultima campanie). Dezechilibrată: ~85% / ~15%
  (1906 / 334), imbalance ratio ~5.7 → `stratify=y`, metrici: balanced accuracy, F1/precision/recall pe clasa 1, ROC AUC.
- **Regresie:** `MntTotal` = suma celor 6 coloane `Mnt*` (creată de noi).
  **Leakage:** scoate toate `Mnt*` din X la regresie.

### Structură / tipuri / probleme găsite
- Tip greșit: `Dt_Customer` e `object` (text "dd-mm-yyyy") → `pd.to_datetime(..., dayfirst=True)`.
- `ID` = coloană de tip ID; `Z_CostContact` (=3), `Z_Revenue` (=11) = constante → se elimină.
- Binare 0/1 (pot rămâne numerice): `AcceptedCmp1–5`, `Complain`, `Response`.
- Categoriale: `Education` (ordinală; "2n Cycle" ≈ Master), `Marital_Status`.
- Valori lipsă: doar `Income`, 24 NaN (~1.1%) → imputare cu mediana în pipeline (doar pe train).
- Coduri ascunse: `Income = 666666`; `Year_Birth` 1893/1899/1900; `Marital_Status` "Absurd", "YOLO", "Alone"(=Single).
- Duplicate: 0 cu ID, ~180 fără ID.
- Asimetrice (skew > 1): `Income` (din cauza outlierului), toate `Mnt*`, `NumDealsPurchases`,
  `NumCatalogPurchases`, `NumWebPurchases`; moderat: `MntTotal`, `NumStorePurchases`, `Kidhome`.

### Curățare aplicată (`df_clean`, salvat în `MY_DIR/clean.csv`)
1. drop `ID`, `Z_CostContact`, `Z_Revenue`
2. `drop_duplicates()`
3. elimină `Year_Birth < 1920` și `Income > 600_000`
4. `Marital_Status`: Alone→Single, Absurd/YOLO→Other
5. `Dt_Customer` → `Customer_Days` (zile de la înscriere până la data maximă), drop `Dt_Customer`
6. adaugă `MntTotal`
- NaN din `Income` rămân → imputate în pipeline.

### Relații cu ținta (aproximativ)
- `Response`: `AcceptedCmp5/1/3`, `MntWines`, `MntMeatProducts`, `MntTotal`, `NumCatalogPurchases` (+);
  `Recency`, `Kidhome`, `Teenhome` (−); clienți vechi răspund mai des; PhD și Single/Divorced/Widow mai des.
- `MntTotal`: `Income` (~0.85), `NumCatalog/Store/WebPurchases` (+); `Kidhome`, `NumWebVisitsMonth` (−).
- Suspecte de leakage: `Mnt*` pentru regresie (sigur); `Num*Purchases` pentru regresie (aceeași perioadă);
  `Mnt*` pentru `Response` (posibil temporal). `AcceptedCmp1–5` NU sunt leakage (campanii anterioare).

## Lab 2 — rezumat

Pași: încărcare → structură (tipuri, cardinalitate, duplicate, constante, ID) → statistici descriptive
(`describe`, `numeric_summary` cu missing%, mean, median, std, CV, min, max, IQR, IQR outliers, skew, kurtosis)
→ valori lipsă + distribuția țintei + statistici per clasă + crosstab → corelații Pearson/Spearman
→ ≥4 plot-uri (titlu = întrebare; include distribuția țintei și o caracteristică vs țintă) → curățare pe copie
→ primele modele (I·C·T).

### Primele modele (I · C · T)
- **I – Isolate:** `train_test_split(test_size=0.2, random_state=SEED)`; `stratify=y` la clasificare;
  date temporale → `shuffle=False` după sortare (nu e cazul aici: `Dt_Customer` e dată de înscriere).
- **C – Construct:** `ColumnTransformer` — numerice: `SimpleImputer(median)` + `StandardScaler`;
  categoriale: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`; totul în `Pipeline`.
- **T – Test:** baseline `DummyClassifier(most_frequent)` / `DummyRegressor(mean)`.
  Modele: LogisticRegression(max_iter=1000), KNeighbors, DecisionTree(max_depth=3) / LinearRegression, KNN, DecisionTreeRegressor.
  Metrici clasificare: accuracy, balanced accuracy, precision, recall, F1 (`pos_label=1`), ROC AUC din `predict_proba[:, 1]`,
  `classification_report`, `ConfusionMatrixDisplay` (nu numi variabila `display`), `tree.plot_tree`.
  Metrici regresie: MAE, RMSE, R².
- Rezultate așteptate: logistică cea mai bună la clasificare (ROC AUC ~0.88–0.90, balanced acc ~0.70, F1 ~0.5);
  baseline acc ~0.85 dar balanced 0.5, F1 0. Regresie: R² ~0.80–0.85 vs 0 la baseline.
- Îmbunătățiri propuse: `class_weight="balanced"`, prag de decizie diferit, feature engineering
  (Age, Children, TotalAcceptedCmp, NumTotalPurchases, log1p), tuning hiperparametri prin CV,
  RandomForest / HistGradientBoosting, ținta `log1p(MntTotal)`.

## Lab 3

Tema: procesele de preprocesare/selecție + validare (detalii pe măsură ce vin). Regula: X_te / y_te
(la mine `X_test` / `y_test`) NU se folosesc până la capitolul 5.

### Pas 1 — încărcare + split
- Exemplul Penguins: citește `penguins_clean.csv` din Drive (altfel reface curățarea din Lab 2);
  scoate `bill_length_mm` și `island` (erau prea predictive, scor 0.99) ca să se vadă efectul proceselor din lab.
- TODO pe setul meu: `df = pd.read_csv(f"{MY_DIR}/clean.csv")` (fallback: refacere curățare Lab 2 din `customer.csv`),
  `TARGET = "Response"`, `X`, `y`, `num_cols`, `cat_cols`; split 80/20 stratificat cu `SEED`,
  nume: `X_train, X_test, y_train, y_test`; afișare shape + `X_train.isna().sum()`.
- Decizie: binarele 0/1 (`AcceptedCmp1–5`, `Complain`) rămân numerice (decizie Lab 2) → `cat_cols` = `Education`, `Marital_Status`.
  Nu scoatem atribute (setul nu are un atribut care rezolvă singur problema; logistica avea ~0.89 acc / ~0.70 balanced).
- Valori lipsă în X_train: doar `Income` (~19–20 din ~1640).

### Pas 2 — preprocesare (exemple Penguins: imputare, scalare cu outlier, kNN cu/fără scalare, OHE/Ordinal, selecție greșită vs corectă, `make_pipeline_ex`)
- Exemplele: `SimpleImputer` mean/median, `add_indicator=True`; categorial `most_frequent` / `constant`;
  Standard/MinMax/Robust pe coloană cu outlier (Robust aproape neafectat, MinMax strivește); kNN fără scalare e dominat de coloana
  cu valori mari; `OneHotEncoder(handle_unknown="ignore")`, `OrdinalEncoder(categories=[[...]])`; `SelectKBest` înainte de CV = leakage
  (≈0.7+ pe zgomot) vs în pipeline (≈0.5); funcția `make_pipeline_ex(model, scaler=None, num_strategy="median")` → pași "prep" și "model".
- TODO pe setul meu → funcția `make_pipeline(model, scaler=None, num_strategy="median", add_indicator=False)`:
  - numerice: mediana (Income și Mnt*/Num* asimetrice);
  - `add_indicator`: nu (doar ~1% lipsă în Income, rata de răspuns similară) — parametru, se poate testa;
  - scaler implicit: `RobustScaler` (outlieri în Mnt*, NumDeals, NumWeb, Income); "passthrough" pentru arbori;
  - `Education` → OrdinalEncoder cu ordinea Basic < 2n Cycle < Graduation < Master < PhD;
  - `Marital_Status` → OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist") (grupează "Other");
  - nicio coloană cu cardinalitate mare.
  - Liste: `ord_cols = ["Education"]`, `nom_cols = ["Marital_Status"]`. Verificare: `pipe.fit(X_train, y_train)` cu LogisticRegression.

### Pas 3 — dezechilibru (doar clasificare)
- Exemplul Penguins: binarizează Chinstrap vs rest; `imblearn.pipeline.Pipeline as ImbPipeline`; strategii none / class_weight /
  RandomOverSampler / SMOTE / RandomUnderSampler; `cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)` comun;
  `cross_validate(..., scoring=["recall","precision","f1","balanced_accuracy"])`. SMOTE înainte de CV = leakage (F1 umflat).
- TODO pe setul meu: `y_train` deja 0/1 (Response); proporție clasa 1 ~0.15, imbalance ratio ~5.7 → dezechilibrat;
  comparăm toate cele 5 strategii cu `prep_bin = make_pipeline(LogisticRegression(max_iter=1000)).named_steps["prep"]`.
- **Rezultate reale** (rulat local pe `lab2/data/raw.csv`, SEED=42; clean 2054 rânduri, train 1643 / test 411,
  clasa 1 = 0.152, ratio 5.57, 1393/250):
  none R=0.440 P=0.711 F1=0.531 BA=0.702 | class_weight R=0.784 P=0.460 F1=0.578 BA=0.808 |
  oversampling R=0.788 P=0.463 F1=0.581 BA=0.811 | SMOTE R=0.796 P=0.461 F1=0.582 BA=0.813 |
  undersampling R=0.784 P=0.428 F1=0.552 BA=0.797. Ales: class_weight.
- Datele brute există local în `lab2/data/raw.csv`; venv-ul proiectului nu are sklearn (am folosit un venv în scratchpad).
### Pas 4 — comparația modelelor prin CV pe X_train
- Exemplul Penguins: CV „de mână” cu `clone` + `cv5.split` = `cross_val_score`; tabel cu val mean/std + train score,
  boxplot pe fold-uri; regresie cu `KFold` și `neg_mean_absolute_error`, `r2`.
- TODO pe setul meu: StratifiedKFold(5, shuffle, SEED); metrici balanced_accuracy, f1, roc_auc; return_train_score=True;
  modele: baseline, logistic(balanced), kNN(5), tree fără limită, tree(max_depth=5), RandomForest, HistGradientBoosting
  (class_weight="balanced" unde există; arbori cu scaler="passthrough"). Salvare: `RESULTS_DIR/L03_{MY_NAME}_cv_results.csv`,
  boxplot în `FIGURES_DIR/L03_{MY_NAME}_cv_boxplot.png`.
- **Rezultate reale (SEED=42)** — val BA mean±std / val F1 / val AUC / train BA:
  baseline 0.500/0.000/0.500/0.500 | logistic 0.808±0.025/0.578/0.898/0.834 | kNN 0.566±0.046/0.230/0.736/0.629 |
  tree no limit 0.648±0.025/0.407/0.648/0.998 | tree d5 0.734±0.047/0.448/0.754/0.836 |
  RF 0.752±0.030/0.553/0.883/0.984 | HGB 0.734±0.023/0.564/0.888/0.998.
  Concluzie: logistica cea mai bună și fără overfitting (train 0.834 vs val 0.808); arborele fără limită, RF, HGB fac overfitting;
  kNN slab (fără class_weight, multe dimensiuni).

### Pas 5 — curba de validare + curba de învățare
- Exemplul Penguins: `validation_curve` pe `model__max_depth` al arborelui, `learning_curve` pe logistică,
  `train_sizes=np.linspace(0.1, 1.0, 6)`; figură 1×2 salvată.
- TODO pe setul meu: validation curve pe `model__C` al logisticii (balanced), C = [0.001, 0.01, 0.1, 1, 10, 100], axă log;
  learning curve pe logistică (cel mai bun model). Salvare: `FIGURES_DIR/L03_{MY_NAME}_curves.png`.
- **Rezultate reale (val BA):** C=0.001 → 0.667 (train 0.675, underfitting); 0.01 → 0.760; 0.1 → 0.789; 1 → 0.808;
  10 → 0.816; 100 → 0.816 (train 0.835). Platou de la C≥1; diferențele 1 vs 10 (0.008) < std (0.025).
- Learning curve: n=131 train 0.899/val 0.739; 367 → 0.855/0.778; 604 → 0.842/0.804; 840 → 0.834/0.799;
  1077 → 0.838/0.809; 1314 → 0.834/0.809. Curbele converg (gap ~0.025) și val se aplatizează de la ~600 → mai multe date
  ajută puțin; pentru câștig trebuie atribute noi / model mai flexibil (model cu bias, nu variance).

### Pas 6 — căutarea hiperparametrilor (GridSearchCV + nested CV)
- Exemplul Penguins: GridSearchCV pe kNN (n_neighbors, weights, prep__num__scale, prep__num__impute__strategy) = 72 combinații;
  `best_params_`, `best_score_` (ușor optimist), top 5 din `cv_results_`; nested CV: inner GridSearch cv=3, outer cv5.
- TODO pe setul meu: 2 modele — regresia logistică (C × class_weight × scaler × impute = 40) și Random Forest
  (max_depth × min_samples_leaf × max_features = 24, class_weight balanced, 300 arbori); scoring balanced_accuracy, cv5.
- **Rezultate reale:** LR best C=1, balanced, StandardScaler, mean/median la egalitate → 0.817 ± 0.028 (train 0.834);
  top 5 toate balanced, C 1–100, diferențe ≤ 0.001. RF best max_depth=10, min_samples_leaf=10, max_features=0.5 → 0.797 ± 0.031
  (train 0.913; față de 0.752 nereglat). Nested CV: LR 0.816 ± 0.028, RF 0.769 ± 0.029 (RF best_score optimist cu ~0.03).
  Câștigător: regresia logistică.

### Pas 7 — evaluarea finală pe test + salvarea modelului (+ prag F1)
- Exemplul Penguins: `best_model = grid.best_estimator_`, o singură evaluare pe test (balanced acc, classification_report,
  ConfusionMatrixDisplay), comparație grid / nested / test; `joblib.dump` în `MODELS_DIR`; prag care maximizează F1 din
  `cross_val_predict(..., method="predict_proba")` pe train.
- TODO pe setul meu: `best_model = grid_lr.best_estimator_` (C=1, balanced, StandardScaler, mean).
- **Rezultate reale pe TEST (411 rânduri, 63 de clasa 1):** BA **0.824**, F1 **0.579**, ROC AUC **0.890**;
  clasa 1: precision 0.442, recall 0.841; clasa 0: precision 0.966, recall 0.807; accuracy 0.813.
  Confusion matrix [[281, 67], [10, 53]]. CV best_score_ 0.817, nested 0.816 → test consistent (+0.007, sub std).
- Prag F1 (OOF pe train): 0.5 → F1 0.583; cel mai bun prag **0.70** → F1 0.624. Pe test cu 0.70: F1 0.593, BA 0.771,
  precision 0.556, recall 0.635, CM [[316, 32], [23, 40]] → trade-off: pragul mare crește precizia, scade recall-ul.
- Salvare: `MODELS_DIR/L03_{MY_NAME}_best.joblib`; rând nou „logistic regression (tuned)” cu test în `L03_{MY_NAME}_cv_results.csv`.

### Concluzii Lab 3 — câștig față de Lab 2 (logistică, același split; CV BA | TEST BA / F1 / AUC / acc)
- Lab 2 (Standard + OHE, fără weight): 0.701 | 0.709 / 0.542 / 0.877 / 0.881
- + preprocesare Lab 3 (Robust, Education ordinal): 0.702 | 0.710 / 0.547 / 0.879 / 0.883 → ~0
- + class_weight balanced: 0.808 | 0.831 / 0.584 / 0.891 / 0.813 → **+0.12 BA, aproape tot câștigul**
- + tuning (StandardScaler, C=1): 0.817 | 0.824 / 0.579 / 0.890 / 0.813 → +0.009 CV, −0.007 test (zgomot)
- Total: BA 0.709 → 0.824 (+0.115), F1 0.542 → 0.579, AUC ~ neschimbat (0.877 → 0.890) → modelul ordona deja bine,
  s-a schimbat doar locul pragului. Acuratețea scade (0.881 → 0.813) — normal la reechilibrare.
- Test vs CV: 0.824 vs 0.817 / nested 0.816 → apropiat, puțin mai mare, în std (~0.028); test mic (63 pozitive) → ±1 client ≈ ±0.008 recall.
- Pași următori: atribute noi (TotalAcceptedCmp, Age, Children, NumTotalPurchases, spend/purchase, log1p), boosting reglat
  (HGB cu learning_rate/max_depth/l2), prag ales după cost, mai multe date nu ajută (learning curve plată).

## Lab 4

_(de completat cu exemplele Penguins și TODO-urile primite; aceeași regulă: soluții pe setul meu, rulate local pe
`lab2/data/raw.csv` cu venv-ul din scratchpad, SEED=42, cifre reale)_

### Pas 0 — setări
- `MY_NAME = "customer"` (presupus: același folder ca `DATA_DIR/customer/` din Lab 1–2), `TARGET_CLF = "Response"`,
  `POS_CLASS = 1` (int, nu "1"), `TARGET_REG = "MntTotal"`, `MY_DIR = f"{DATA_DIR}/{MY_NAME}"`.
- Lab 4 are DOUĂ ținte simultan (clasificare + regresie).

### Pas 1 — cele două probleme pe același tabel
- Exemplul Penguins: clasificare binară „Chinstrap?”, regresie body_mass_g (specia devine atribut), două split-uri,
  `cv5_c = StratifiedKFold`, `cv5_r = KFold`; X_te/y_te nu se folosesc în lab.
- TODO pe setul meu (nume: Xc_train…, Xr_train…, num_c/cat_c, num_r/cat_r, cv_c, cv_r):
  - clasificare: X fără `Response` și fără `MntTotal` (suma exactă a celor 6 Mnt* → redundantă/coliniară; Mnt* rămân);
    `yc = (df[TARGET_CLF] == POS_CLASS).astype(int)`; split stratificat.
  - regresie: X fără `MntTotal`, cele 6 `Mnt*` (leakage) și `Response` (rezultatul unei campanii viitoare, nu profil);
    `Num*Purchases` rămân (suspecte, discutate în Lab 2); split simplu, fără stratify.
  - nu e nevoie de split cronologic (Customer_Days = vechime, nu serie de timp).
- **Cifre reale:** clf train (1643, 25) / test (411, 25), clasa 1 = 0.152, 23 NaN (Income);
  reg train (1643, 19) / test (411, 19), MntTotal train mean 598.7, median 393.0, skew 0.89 (test mean 637.4).

### Pas 2 — funcțiile ajutătoare (din exemplu, folosite pe setul meu)
- `make_pipeline_ex(model, num_cols, cat_cols, scaler=None, num_strategy="median")` (Standard + OHE),
  `scoring_for(task, y)` (binar: PR_AUC, ROC_AUC, F1, recall — prima = principală; regresie: RMSE, MAE, R2),
  `run_cv(pipe, X, y, task, cv)` → (row, main_folds), `corrected_ttest(a, b, n_train, n_test)` (Nadeau-Bengio).
- Eroare la rulare: ultima parte a exemplului folosește numele Penguins (`Xc_tr`, `yc_tr`, `cv5_c`, `Xr_tr`, `yr_tr`, `cv5_r`)
  → NameError; pe setul meu: `Xc_train, yc_train, cv_c, Xr_train, yr_train, cv_r`.
- **Verificare reală:** LogReg (fără class_weight) PR_AUC 0.638 ± 0.042 (train 0.692), ROC_AUC 0.895, F1 0.531, recall 0.444;
  LinearRegression RMSE 263.5 ± 10.5 (train 253.8), MAE 177.2, R² 0.803.

### Pas 3 — funcția mea de evaluare `evaluate(model, X, y, cv, task, name=None)` → (summary dict, folds)
- `METRICS`: clasificare `PR_AUC` (average_precision, principală) + `bal_acc` + `recall`;
  regresie `MAE` (principală) + `RMSE` + `R2`. Prima cheie = principală (mean, std, train, gap; gap = cât e train mai bun decât val,
  semn corectat pentru erori). Folds = scorurile principale pe fold-uri (pentru corrected_ttest).
- De ce: PR_AUC nu depinde de prag și se uită doar la clasa rară (baseline = 0.152); bal_acc pentru continuitate cu Lab 3;
  recall = ce contează la campanie. MAE în unități monetare, robustă la clienții care cheltuie mult (ținta asimetrică).
- **Verificare reală:** clf baseline PR_AUC 0.152, bal_acc 0.500; logistic balanced PR_AUC 0.637 ± 0.039 (train 0.667, gap 0.030),
  bal_acc 0.812, recall 0.796; folds [0.627, 0.643, 0.650, 0.693, 0.571].
  reg baseline MAE 507.4 ± 12.7, R² ≈ 0; linear MAE 177.2 ± 7.5 (train 172.1, gap 5.1), RMSE 263.5, R² 0.803.

### Setul de IMAGINI atribuit: I8 Fashion-MNIST (Keras) — pereche cu T8 (Customer Personality Analysis)
- Poveste: lanțul de magazine își digitalizează catalogul; fiecare produs fotografiat trebuie pus automat în categoria
  potrivită (tricou, pantof, geantă…) ca să apară în campania de oferte potrivită. Domeniu: retail / marketing.
- Sarcină: clasificare, tipul produsului, **10 clase**. 70.000 imagini alb-negru **28×28** (~30 MB):
  60.000 train + 10.000 „val” (setul de test oficial). Sursa: https://keras.io/api/datasets/fashion_mnist/
- Încărcare (după celula comună, care importă `keras`):
  `(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()`
  `class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`
- Date: `uint8` 0–255, forma (60000, 28, 28); etichete 0–9; clase perfect echilibrate (6000/clasă în train).
  Normalizare tipică `/ 255.0`; pentru modele sklearn/MLP: `reshape(-1, 784)`; pentru CNN: `[..., None]` → (N, 28, 28, 1).
- **Atenție la nume:** `x_train`/`y_train` (imagini, litera mică) se pot suprapune cu `y_train` din partea tabelară →
  dacă sunt în același notebook, folosește nume distincte (ex. `xi_train`, `yi_train`) sau ordinea celulelor contează.
- Clase confundate frecvent: Shirt ↔ T-shirt/top ↔ Pullover ↔ Coat; Sneaker ↔ Sandal ↔ Ankle boot.

## Note vechi

- (estimare inițială pas 3 Lab 3) none → recall ~0.4, precision ~0.65–0.7; class_weight / SMOTE / oversampling → recall ~0.75–0.8,
  precision ~0.4, F1 similar sau puțin mai mare; balanced accuracy crește de la ~0.70 la ~0.80. Alegere: class_weight
  (cel mai simplu, fără puncte sintetice) dacă recall-ul e important (campanie de marketing = ratarea unui client interesat costă mai mult).
