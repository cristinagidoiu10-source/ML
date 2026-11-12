# Rezultate și concluzii – Proiectul A

## Cele mai importante 7 lucruri din proiect

1. **Problema e una de business reală:** magazinul vrea să trimită oferta doar clienților care au șanse să o accepte → costuri mai mici, clienți mai puțin deranjați.
2. **Clasele sunt dezechilibrate (15% / 85%)**, deci **acuratețea minte**: un model care spune mereu „nu” are 85% acuratețe și e inutil. Am folosit ROC-AUC, recall, precizie, F1 și PR-AUC.
3. **Tratarea dezechilibrului cu `class_weight="balanced"`** a crescut recall-ul de la **0,50 la 0,80** (găsim 80% din clienții interesați în loc de 50%).
4. **Zero data leakage:** testul a fost pus deoparte de la început și folosit o singură dată; toată preprocesarea a stat într-un pipeline; scorul din CV (0,90) s-a confirmat pe test (0,89).
5. **Modelul ales e cel mai simplu – Regresia Logistică.** A egalat SVM-ul și a bătut Random Forest la recall, dar e explicabilă. Principiu: *nu complici modelul dacă nu câștigi nimic*.
6. **Am demonstrat cu experimente** că deciziile au contat: fără scalare SVM-ul cade de la 0,90 la 0,62 ROC-AUC; fără `class_weight` recall-ul scade la jumătate.
7. **Rezultatele sunt stabile și credibile:** pe 10 împărțiri diferite ROC-AUC = 0,892 ± 0,010; duplicatele din date nu au umflat scorul.

## Întrebarea 1 – Cine va accepta oferta?

**Răspuns: da, se poate prezice destul de bine.**

- Model: **Regresia Logistică** (C = 1, `class_weight="balanced"`).
- Pe test: **ROC-AUC 0,892**, **recall 0,761** (găsește 51 din 67 de clienți care au răspuns), precizie 0,44, F1 0,557.
- SVM (RBF, C = 10, gamma = 0,001) a obținut aceleași scoruri, dar e mai greu de explicat.
- Random Forest are precizia cea mai mare (0,60), dar **ratează jumătate** din clienții interesați.

**Profilul clientului care răspunde:** a cumpărat recent, e client vechi, cheltuie mult (mai ales pe carne), cumpără din catalog, a mai acceptat campanii, e mai des necăsătorit / divorțat / văduv.

## Întrebarea 2 – Cât cheltuie un client?

**Răspuns: da, în mare parte.**

- Model: **Random Forest Regressor**.
- Pe test: eroare medie (**MAE**) **~108** la o cheltuială medie de ~605; **R² = 0,88** (explică 88% din diferențe).
- Modelul liniar (Ridge) e mai slab (R² 0,80) → aici relația nu e liniară și ansamblul de arbori câștigă.

## Ce a arătat analiza hiperparametrilor

| Model | Comportament |
|---|---|
| Regresia Logistică | stabilă pentru orice C ≥ 0,1; train ≈ validare → nu face overfitting |
| SVM | foarte sensibil la `gamma`: prea mare (1) → memorează datele, scor 0,78 |
| Random Forest | ajunge la 1,0 pe train (overfitting), fără câștig pe validare după `max_depth` = 10 |

## Limitări (spuse cinstit)

- Setul de test e mic: doar **67** de clienți care au răspuns → un client în plus sau în minus schimbă recall-ul cu ~1,5%. De aceea am făcut 10 rulări.
- Datele sunt dintr-un singur magazin, din 2012–2014.
- La regresie, numărul de cumpărături e măsurat în aceeași perioadă ca cheltuiala → problema e puțin „ușurată”.
- Pragul de decizie a rămas cel implicit (0,5); ar putea fi ales după costul real al unui contact.

## Ce s-ar putea face în continuare

- alegerea **pragului de decizie după profit** (costul unui contact vs. câștigul unui client);
- un model de **gradient boosting**;
- transformarea logaritmică a țintei de regresie (cheltuiala e foarte asimetrică).

## Tabel-rezumat cu toate cifrele cheie

| Ce | Valoare |
|---|---|
| Clienți (după curățare) | 2236 (din 2240) |
| Train / test | 1788 / 448 |
| Procent clienți care răspund | 14,9% |
| Venituri lipsă | 24 |
| Coloane după preprocesare | 38 |
| ROC-AUC Regresie Logistică (CV / test) | 0,904 / 0,892 |
| Recall Regresie Logistică (test) | 0,761 |
| ROC-AUC Dummy | ~0,5 |
| ROC-AUC pe 10 împărțiri | 0,892 ± 0,010 |
| SVM fără scalare | ROC-AUC 0,623 |
| Recall fără class_weight (SVM) | 0,371 (vs. 0,779) |
| Regresie: MAE / R² (Random Forest) | 108,1 / 0,883 |
