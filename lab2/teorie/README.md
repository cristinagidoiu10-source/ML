# Teorie — laboratoarele ML pe setul Customer Personality Analysis

Fișierele de teorie din [Dicoi12/ML-LAB](https://github.com/Dicoi12/ML-LAB) (acolo pe Telco Customer Churn), refăcute pe **setul nostru** (`lab2/data/raw.csv`, 2240 de clienți). Fiecare fișier are aceeași structură ca în repo: termenul → **ce înseamnă** → **la noi** (cu cifrele noastre) → **în cod**, iar la final o secțiune **„Codul din Lab N, explicat”**, cu codul laboratorului luat bucată cu bucată.

| Fișier | Ce conține | Echivalent în repo |
|---|---|---|
| [00_COLOANE_SET_DATE.md](00_COLOANE_SET_DATE.md) | fiecare coloană explicată, problemele de date, rolul fiecărei coloane | — (nou) |
| [TEORIE_LAB2.md](TEORIE_LAB2.md) | încărcare, structură, statistici, corelații, leakage, curățare, primele modele + `todo1/2/3.py` explicate | `lab2/TEORIE_LAB2.md` |
| [TEORIE_LAB3.md](TEORIE_LAB3.md) | pipeline, imputare, scalare, codificare, dezechilibru, CV, curbe, GridSearch, nested CV, test | `lab3/TEORIE_LAB3.md` |
| [TEORIE_LAB4.md](TEORIE_LAB4.md) | SVM / SVR, C, kernel, γ, ε, calibrare, prag, reziduuri, test t corectat | `lab4/TEORIE_LAB4.md` |
| [TEORIE_LAB5.md](TEORIE_LAB5.md) | bagging, random forest, boosting, voting/stacking, selecție, test final, lift, importanță, profit | `lab5/TEORIE_LAB5.md` |
| [TEORIE_LAB6.md](TEORIE_LAB6.md) | rețele neuronale cu Keras vs. scikit-learn | `lab6/TEORIE_LAB6.md` |
| [TEORIE_CUSTOMER.md](TEORIE_CUSTOMER.md) | rezumatul tuturor deciziilor și rezultatelor, pe o pagină | `TEORIE_TELCO.md` |
| [TEORIE_INTERACTIVA.html](TEORIE_INTERACTIVA.html) | toată teoria explicată simplu, cu exemple interactive (prag, kNN, gamma, overfitting, calibrare…) și întrebări de examen; se deschide în browser | — (nou) |
| [LAB4_INTERACTIV.html](LAB4_INTERACTIV.html) | Lab 4 interactiv: SVM liniar antrenat în pagină (C), harta RBF (γ), tubul ε, scalare, calibrare, prag OOF, reziduuri, testul t corectat | — (nou) |
| [LAB5_INTERACTIV.html](LAB5_INTERACTIV.html) | Lab 5 interactiv: calculator de vot, bootstrap / OOB, random forest, boosting (η), voting, log pe țintă, căutare aleatoare, IC bootstrap, lift, importanță, profit | — (nou) |
| [LAB6_INTERACTIV.html](LAB6_INTERACTIV.html) | Lab 6 interactiv: neuronul, activări, numărul de parametri, pierderi, coborârea pe gradient, curbe de antrenare + EarlyStopping, Dropout, class_weight, seed-uri | — (nou) |

**Ordinea de citit:** `00_COLOANE` → `TEORIE_LAB2` → … → `TEORIE_LAB6`, iar `TEORIE_CUSTOMER` ca recapitulare înainte de prezentare.

**De unde vin cifrele:**
- Lab 2: calculate direct pe `raw.csv` cu `python lab2\todo3.py` (fișier nou, echivalentul lui `todo3.py` din repo);
- Lab 3–5: din codul corectat din `L03_raport.md`, `L04_raport.md`, `L05_raport.md` (rulat pe `clean.csv`, scikit-learn 1.9). Pe Colab pot diferi la a treia zecimală;
- Lab 6: rețelele depind de seed și hardware; fișierul dă cifrele fixe (dimensiuni, parametri, ponderi) și explică ce să cauți în rezultatele tale.
