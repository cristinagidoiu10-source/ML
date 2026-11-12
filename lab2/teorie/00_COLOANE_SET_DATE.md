# Coloanele setului de date — Customer Personality Analysis

Setul: [Kaggle — imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), fișierul `marketing_campaign.csv` (separator TAB), copiat local în `lab2/data/raw.csv` (separator virgulă).

**2240 de rânduri × 29 de coloane. Un rând = un client** al unui lanț de magazine care vinde în magazin fizic, online și prin catalog. Datele acoperă clienți înscriși între **30 iulie 2012** și **29 iunie 2014**; sumele sunt cheltuite în **ultimii 2 ani**.

**Povestea:** firma pregătește o nouă campanie de oferte.
- **Clasificare:** ce clienți vor accepta oferta? → ținta `Response` (0/1).
- **Regresie:** cât cheltuie un client în 2 ani? → ținta `MntTotal` = suma celor 6 coloane `Mnt*` (o creăm noi).

Toate cifrele de mai jos sunt calculate pe `raw.csv` (le poți reproduce cu `python lab2\todo3.py`).

---

## Un client citit ca o poveste (rândul 1, ID 5524)

| Coloană | Valoare | Cum se citește |
|---|---|---|
| `Year_Birth` | 1957 | născut în 1957 → ~57 de ani în 2014 |
| `Education` | Graduation | are licență |
| `Marital_Status` | Single | singur |
| `Income` | 58138 | venit anual al gospodăriei 58.138 |
| `Kidhome` / `Teenhome` | 0 / 0 | nu are copii acasă |
| `Dt_Customer` | 04-09-2012 | client din 4 septembrie 2012 |
| `Recency` | 58 | ultima cumpărare acum 58 de zile |
| `MntWines`…`MntGoldProds` | 635, 88, 546, 172, 88, 88 | în 2 ani: 1617 cheltuiți |
| `NumDeals/Web/Catalog/StorePurchases` | 3, 8, 10, 4 | 22 de cumpărături, 3 cu reducere |
| `NumWebVisitsMonth` | 7 | intră pe site de ~7 ori pe lună |
| `AcceptedCmp1..5` | toate 0 | n-a acceptat nicio campanie anterioară |
| `Complain` | 0 | nu s-a plâns |
| `Response` | **1** | **a acceptat ultima ofertă** |

---

## 1. Identificare și constante (se elimină)

| Coloană | Tip | Ce înseamnă | Ce facem |
|---|---|---|---|
| `ID` | int | numărul clientului. 2240 de valori unice din 2240 de rânduri → **identificator** | se scoate: nu descrie clientul, modelul ar memora ID-uri |
| `Z_CostContact` | int | costul contactării unui client în campanie. Mereu **3** → **constantă** | se scoate: cardinalitate 1, zero informație |
| `Z_Revenue` | int | venitul adus de un client care acceptă. Mereu **11** → **constantă** | se scoate (dar e utilă pentru *metrica de business*: profit = 11 × acceptări − 3 × contactări) |

## 2. Cine e clientul (date demografice)

| Coloană | Tip | Ce înseamnă | Valori / observații |
|---|---|---|---|
| `Year_Birth` | numerică (int) | anul nașterii | 1940–1996, plus **3 valori imposibile: 1893, 1899, 1900** (vârste 114–121 de ani) → eroare de introducere. O transformăm în `Age = 2014 − Year_Birth` (mediana 44 de ani) |
| `Education` | categorială **ordinală** | nivelul de studii | Graduation 1127, PhD 486, Master 370, 2n Cycle 203, Basic 54. Ordinea: Basic < Graduation < 2n Cycle ≈ Master < PhD. **„2n Cycle”** = ciclul 2 Bologna = practic master |
| `Marital_Status` | categorială **nominală** | starea civilă | Married 864, Together 580, Single 480, Divorced 232, Widow 77 + **coduri ascunse**: Alone 3 (= Single), Absurd 2, YOLO 2 (răspunsuri fără sens → „Other”) |
| `Income` | numerică (float) | venitul anual al gospodăriei | mediana 51.382; **24 de valori lipsă** (1,07%) → imputare cu mediana; **666666** = valoare-santinelă (cifre repetate), aproape de 4 ori mai mare decât următorul venit (162.397) → eroare |
| `Kidhome` | numerică discretă | câți copii mici are acasă | 0, 1, 2. Rămâne numerică: e o numărătoare, are ordine și distanțe reale |
| `Teenhome` | numerică discretă | câți adolescenți are acasă | 0, 1, 2 |

## 3. Relația cu magazinul

| Coloană | Tip | Ce înseamnă | Observații |
|---|---|---|---|
| `Dt_Customer` | **text** (`object`) — tip greșit | data înscrierii ca client, format `zi-lună-an` | 30.07.2012 – 29.06.2014. O convertim cu `pd.to_datetime(..., format="%d-%m-%Y")` și apoi în `Customer_Days` = câte zile e client (0–699) |
| `Recency` | numerică | câte **zile** au trecut de la ultima cumpărare | 0–99, uniform distribuită (media 49). Clienții care au cumpărat **recent** răspund mai des (mediana 30 la cei care acceptă vs 52) |
| `Complain` | binară 0/1 | s-a plâns în ultimii 2 ani? | doar **21** de clienți (0,9%) → aproape constantă; rata de răspuns e aceeași (14,3% vs 14,9%), deci aproape inutilă |

## 4. Cât a cheltuit în ultimii 2 ani — coloanele `Mnt*` („Mnt” = *amount*)

| Coloană | Pe ce | Media | Mediana | Max | Din total |
|---|---|---|---|---|---|
| `MntWines` | vin | 304 | 174 | 1493 | **50%** |
| `MntMeatProducts` | carne | 167 | 67 | 1725 | **28%** |
| `MntGoldProds` | aur / produse premium | 44 | 24 | 362 | 7% |
| `MntFishProducts` | pește | 38 | 12 | 259 | 6% |
| `MntSweetProducts` | dulciuri | 27 | 8 | 263 | 4,5% |
| `MntFruits` | fructe | 26 | 8 | 199 | 4% |
| **`MntTotal`** (creată) | **suma celor 6** = ținta de regresie | **606** | **396** | 2525 | 100% |

- Toate sunt **puternic asimetrice spre dreapta** (skew 1,2–2,1): mulți clienți cheltuie puțin, puțini foarte mult. Media ≫ mediana.
- **Leakage la regresie:** suma lor **este** ținta, deci la regresie **toate** `Mnt*` ies din X.

## 5. Cum cumpără — coloanele `Num*`

| Coloană | Ce numără | Media | Max |
|---|---|---|---|
| `NumDealsPurchases` | cumpărături făcute **cu reducere** | 2,3 | 15 |
| `NumWebPurchases` | cumpărături **online** | 4,1 | 27 |
| `NumCatalogPurchases` | cumpărături **din catalog** | 2,7 | 28 |
| `NumStorePurchases` | cumpărături **în magazin** | 5,8 | 13 |
| `NumWebVisitsMonth` | vizite pe site **în ultima lună** (nu cumpărături) | 5,3 | 20 |

- `NumCatalogPurchases` e cea mai legată de cheltuială (Spearman **0,89** cu `MntTotal`) și una dintre cele mai legate de `Response` (0,23).
- **Leakage la regresie:** numărul de cumpărături e numărat în **aceeași perioadă de 2 ani** ca `MntTotal` (corelație 0,91). Pentru un client nou nu le știm → la regresie le scoatem și pe ele. `NumWebVisitsMonth` e corelată **negativ** (−0,48): cine intră des pe site cumpără mai puțin (se uită la oferte, are venit mai mic).

## 6. Campaniile de marketing

| Coloană | Ce înseamnă | Câți au acceptat | Rata de răspuns la ultima campanie, dacă au acceptat-o pe aceasta |
|---|---|---|---|
| `AcceptedCmp1` | a acceptat campania 1? (0/1) | 144 | **54,9%** (vs 12,2%) |
| `AcceptedCmp2` | campania 2 | 30 | **66,7%** (vs 14,2%) |
| `AcceptedCmp3` | campania 3 | 163 | 47,2% (vs 12,4%) |
| `AcceptedCmp4` | campania 4 | 167 | 37,1% (vs 13,1%) |
| `AcceptedCmp5` | campania 5 | 163 | **56,4%** (vs 11,7%) |
| **`Response`** | **a acceptat ULTIMA campanie (a 6-a)?** → **ținta de clasificare** | **334 (14,9%)** | — |

- Ordinea coloanelor în fișier e ciudată (3, 4, 5, 1, 2), dar numele spune campania.
- `AcceptedCmp1–5` **nu** sunt leakage: sunt campanii **anterioare**, le știm înainte de cea nouă. Sunt cei mai buni predictori (`AcceptedCmp5` Spearman 0,33).
- Clasele sunt **dezechilibrate**: 85,1% nu / 14,9% da, raport **5,71** → `stratify=y`, `class_weight="balanced"`, metrici F1 / PR-AUC, nu acuratețe.

---

## Rezumat: ce rol are fiecare coloană

| Rol | Coloane |
|---|---|
| eliminate | `ID`, `Z_CostContact`, `Z_Revenue` |
| transformate | `Year_Birth` → `Age`, `Dt_Customer` → `Customer_Days` |
| categoriale | `Education` (ordinală), `Marital_Status` (nominală) |
| numerice | `Age`, `Income`, `Kidhome`, `Teenhome`, `Recency`, `Customer_Days`, `Mnt*`, `Num*` |
| binare 0/1 | `AcceptedCmp1–5`, `Complain` |
| ținta de clasificare | `Response` |
| ținta de regresie | `MntTotal` (creată) |
| interzise la regresie (leakage) | toate `Mnt*`, `NumDeals/Web/Catalog/StorePurchases`, `Response` |
| interzise la clasificare | `MntTotal` (e doar suma `Mnt*`, care rămân deja în X) |

## Problemele de date găsite (le curățăm în Lab 2, TODO 5)

| Problemă | Unde | Câte | Decizie |
|---|---|---|---|
| valori lipsă | `Income` | 24 | mediana, **în pipeline** (învățată doar pe train) |
| valoare-santinelă | `Income = 666666` | 1 | se elimină rândul (sau → NaN) |
| an imposibil | `Year_Birth < 1920` | 3 | se elimină rândurile |
| categorii fără sens | `Marital_Status` Absurd / YOLO | 4 | → „Other” |
| sinonim | `Marital_Status` Alone | 3 | → Single |
| tip greșit | `Dt_Customer` (text) | toată coloana | → dată → `Customer_Days` |
| duplicate ascunse de ID | rânduri identice fără `ID` | 182 | se elimină (același client introdus de două ori) |
| constante | `Z_CostContact`, `Z_Revenue` | 2 coloane | se elimină |
