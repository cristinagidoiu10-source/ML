# Teorie + deciziile pentru Customer Personality (laboratoarele 2–6)

> Echivalentul lui `TEORIE_TELCO.md` din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB), pe setul nostru. E **rezumatul** pe o pagină: ce am decis la fiecare pas și de ce. Detaliile și codul explicat sunt în fișierele pe laboratoare:
> [00_COLOANE_SET_DATE](00_COLOANE_SET_DATE.md) · [TEORIE_LAB2](TEORIE_LAB2.md) · [TEORIE_LAB3](TEORIE_LAB3.md) · [TEORIE_LAB4](TEORIE_LAB4.md) · [TEORIE_LAB5](TEORIE_LAB5.md) · [TEORIE_LAB6](TEORIE_LAB6.md)

---

## 1. Setul de date și cele două probleme

- **2240 de clienți × 29 de coloane** ai unui lanț de magazine (magazin, online, catalog), înscriși 2012–2014.
- **Clasificare:** `Response` — acceptă clientul oferta din noua campanie? 14,9% da (334), raport de dezechilibru 5,71.
- **Regresie:** `MntTotal` = suma celor 6 coloane `Mnt*` — cât cheltuie clientul în 2 ani? Media 606, mediana 396, skew 0,86.
- **Povestea:** fiecare contact costă 3 (`Z_CostContact`), fiecare acceptare aduce 11 (`Z_Revenue`). Să contactezi pe toată lumea pierde 1,33 pe client; modelul trebuie să aleagă **cui** trimitem oferta.

## 2. Ce am găsit la explorare (Lab 2)

| Constatare | Cifre |
|---|---|
| identificator / constante | `ID`; `Z_CostContact` = 3, `Z_Revenue` = 11 |
| duplicate ascunse de ID | 182 |
| valori lipsă | `Income`: 24 (1,07%) |
| coduri ascunse | `Income = 666666`; `Year_Birth` 1893/1899/1900; `Marital_Status` Absurd, YOLO, Alone; `Education` 2n Cycle = Master |
| tip greșit | `Dt_Customer` (text `zi-lună-an`) |
| cele mai legate de `Response` | campaniile anterioare (`AcceptedCmp5` 56% vs 12% rată de răspuns), `Recency` (−), `MntTotal`, `NumCatalogPurchases`, `Customer_Days` |
| cele mai legate de `MntTotal` | `NumCatalogPurchases` 0,89, `Income` 0,85, `Kidhome` −0,62 |
| leakage la regresie | `Mnt*` (sunt ținta), `Num*Purchases` (aceeași perioadă, 0,91), `Response` |

## 3. Curățarea (Lab 2)

Scoatem `ID`, constantele și duplicatele; `666666` și anii imposibili → `NaN`; categoriile ciudate unificate; `Dt_Customer` → `Customer_Days` (față de 31.12.2014); construim `MntTotal`. **Nu** completăm lipsurile (se face în pipeline, pe train). `(2240, 29)` → **`(2058, 27)`**.

## 4. Preprocesarea (Lab 3)

| Coloane | Transformare | De ce |
|---|---|---|
| numerice | `SimpleImputer(median)` + `StandardScaler` / `RobustScaler` | asimetrie, outlieri; scalarea e obligatorie la logistică / kNN / SVM |
| `Education` | `OrdinalEncoder(Basic < Graduation < Master < PhD)` | are ordine |
| `Marital_Status` | `OneHotEncoder(min_frequency=10)` | fără ordine; „Other” (4) → infrequent |
| `AcceptedCmp*`, `Complain`, `Kidhome`, `Teenhome` | rămân numerice | 0/1 sau numărători |
| arbori | `scaler="passthrough"` | nu au nevoie de scalare |

Totul într-un `Pipeline` → învățat doar pe train (în CV: pe fiecare fold).

## 5. Metrici

| Problemă | Principală | Secundare | De ce |
|---|---|---|---|
| clasificare (L03) | acuratețe echilibrată | F1, ROC-AUC | baseline 0,5, nu 0,85 |
| clasificare (L04+) | **PR-AUC** | F1, recall | măsoară ordonarea clienților; baseline 0,15 |
| regresie | **MAE** | RMSE, R² | „greșim în medie cu X”; RMSE ar fi dominat de clienții mari |

## 6. Dezechilibrul și pragul

- `class_weight="balanced"` (≈ 0,59 / 3,3) → recall 0,41 → **0,78** la regresia logistică; la egalitate cu oversampling / SMOTE, dar cel mai simplu.
- Pragul se alege pe **predicții out-of-fold** din train: cea mai mare precizie cu **recall ≥ 0,75** (L04: 0,19 pe SVM calibrat; L05: **0,53** pe regresia logistică).
- Pragul după **profit** (L05) = 0,59 = pragul după F1.

## 7. Validarea

5 fold-uri (`StratifiedKFold` / `KFold`, `shuffle=True`, același `SEED`) pentru **toate** modelele; medie ± std; diferențe credibile doar peste o std; **testul t corectat** (Nadeau–Bengio) pentru „e semnificativ?”; nested CV pentru estimarea căutării; testul **o singură dată**.

## 8. Modelele, pe scurt (CV pe train)

| Model | Clasificare PR-AUC | Regresie MAE |
|---|---|---|
| baseline | 0,152 | 527 |
| regresie logistică / liniară | **0,601** | 233 |
| kNN | slab (fără `class_weight`) | — |
| arbore | 0,324 | 233,5 |
| random forest | 0,566 | 192,2 |
| extra-trees | 0,584 | 187,3 |
| SVM / SVR (RBF) | 0,606 | 199,1 |
| GradientBoosting | 0,598 (recall mic) | 182,7 |
| HistGB | 0,553 → reglat 0,603 | **185,5** |
| voting | **0,630** | 193,0 |
| stacking | 0,625 | 186,1 |
| rețea Keras (L06) | ≈ la nivelul logisticii, mai lentă | ≈ la nivelul HistGB |

**Testul t:** voting vs regresia logistică p = 0,275; SVM vs logistică p = 0,845 → **nesemnificativ** → la scor egal alegem modelul **mai simplu**.

## 9. Modelele finale și testul (Lab 5)

| | Model | CV | **Test** | IC 95% |
|---|---|---|---|---|
| clasificare | regresie logistică, `class_weight="balanced"`, C = 1, prag 0,53 | PR-AUC 0,601 | **0,744** (recall 0,857, precizie 0,535, ROC-AUC 0,917) | 0,633 – 0,857 |
| regresie | HistGB (implicit, oprire timpurie) | MAE 185,5 | **160,0** (R² 0,826) | 144,7 – 176,0 |

- **Lift@20% = 3,97:** primii 20% clienți după scor conțin 79% dintre cei care acceptă.
- **Profit:** de la −1,33 pe client (contactăm pe toți) la **+0,52** (train OOF) / **+0,74** (test) pe client.

## 10. Ce contează (interpretare)

- **Cine acceptă oferta:** clienții care au cumpărat **recent** (`Recency`), care au **acceptat campanii anterioare** (`AcceptedCmp5`, `AcceptedCmp3`), care cheltuie pe **carne / vin** și cumpără din **catalog**, clienți **vechi**, **fără adolescenți**, **singuri / divorțați / văduvi** (cuplurile răspund de 2 ori mai rar).
- **Cât cheltuie:** aproape totul e **venitul** (`Income` — MAE crește cu 366 fără el), apoi vechimea și copiii (care scad cheltuiala).
- **Unde greșește regresia:** subestimează clienții foarte mari (pâlnie în reziduuri); venitul mare nu garantează cheltuieli mari.

## 11. Ce n-a ajutat (și de ce e important să o spunem)

- transformarea log a țintei (skew doar 0,87; arborii sunt insensibili);
- selecția atributelor (doar ~25 de coloane; orice eliminare costă);
- stacking-ul (+0,02, sub zgomot, de 10–100 de ori mai lent);
- rețelele neuronale (tabel mic → egalează, dar sunt lente și instabile).

## 12. Toate deciziile, într-un tabel

| # | Decizie | Lab |
|---|---|---|
| 1 | ținte: `Response` (clasificare), `MntTotal` (regresie, construită) | 2 |
| 2 | scoatem `ID`, `Z_*`, 182 de duplicate | 2 |
| 3 | `666666`, anii < 1920 → `NaN`; categoriile ciudate unificate | 2 |
| 4 | `Dt_Customer` → `Customer_Days` (dată fixă) | 2 |
| 5 | la regresie fără `Mnt*`, `Num*Purchases`, `Response` (leakage) | 2, 4 |
| 6 | split 80/20, stratificat la clasificare, aleator (nu cronologic) | 2, 3 |
| 7 | preprocesarea în pipeline; mediana; ordinal + one-hot | 3 |
| 8 | `class_weight="balanced"` | 3 |
| 9 | 5-fold CV, aceleași fold-uri, test t corectat | 3, 4 |
| 10 | metrici: PR-AUC / MAE, alese înainte | 4 |
| 11 | prag pe OOF, recall ≥ 0,75 | 4, 5 |
| 12 | la scor egal → modelul mai simplu | 4, 5 |
| 13 | final: regresie logistică + HistGB | 5 |
| 14 | fără log pe țintă, fără selecție | 5 |
| 15 | rețeaua nu merită pe acest set | 6 |
