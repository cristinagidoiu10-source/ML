# Setul de date, pe înțelesul meu

> Notițe personale, nu fac parte din laborator. Scopul lor: să înțeleg ce e în date înainte să lucrez cu ele.

## Ce e, pe scurt

Un tabel cu **2240 de clienți** ai unui lanț de magazine (un fel de supermarket care vinde și prin catalog și online). **Un rând = un client.** Pentru fiecare client știm:

1. **cine e** — vârstă, studii, stare civilă, venit, copii;
2. **cât a cheltuit în ultimii 2 ani**, pe categorii de produse;
3. **cum cumpără** — din magazin, online, din catalog, cu reducere;
4. **cum a reacționat la campaniile de marketing** — a acceptat sau nu ofertele.

Datele sunt din jurul anilor **2012–2014** (clienții s-au înscris între iulie 2012 și iunie 2014). Moneda nu e precizată (cel mai probabil dolari).

## Un client, citit ca o poveste

Primul rând din tabel (ID 5524):

> Născut în **1957**, are o licență (**Graduation**), e **Single**, câștigă **58.138 pe an** și **nu are copii** acasă. E client din **4 septembrie 2012**, iar ultima dată a cumpărat acum **58 de zile**.
>
> În 2 ani a cheltuit **1617**: 635 pe vin, 546 pe carne, 172 pe pește, câte 88 pe fructe, dulciuri și aur/bijuterii.
>
> A cumpărat de **4 ori din magazin**, de **8 ori online** și de **10 ori din catalog**, iar 3 dintre cumpărături au fost cu reducere. Intră pe site de vreo **7 ori pe lună**.
>
> N-a acceptat niciuna dintre primele 5 campanii, dar **a acceptat-o pe ultima** (`Response = 1`). N-a făcut reclamații.

Fiecare rând se citește la fel.

## Coloanele, pe grupuri

### 1. Cine e clientul
| Coloană | Ce înseamnă | Exemplu / interval |
|---|---|---|
| `ID` | numărul clientului (o etichetă, nu o informație) | 0 … 11191 |
| `Year_Birth` | anul nașterii | 1940 … 1996 (plus 3 valori imposibile: 1893, 1899, 1900) |
| `Education` | studii | Basic, Graduation (licență), 2n Cycle (≈ master), Master, PhD |
| `Marital_Status` | stare civilă | Married, Together, Single, Divorced, Widow (+ ciudățenii: Alone, Absurd, YOLO) |
| `Income` | venit anual al gospodăriei | median ≈ 51.400; lipsește la 24 de clienți |
| `Kidhome` | câți copii mici are acasă | 0, 1, 2 |
| `Teenhome` | câți adolescenți are acasă | 0, 1, 2 |

### 2. Relația cu magazinul
| Coloană | Ce înseamnă |
|---|---|
| `Dt_Customer` | când s-a înscris ca client (scris ca text `zi-lună-an`) |
| `Recency` | câte zile au trecut de la ultima cumpărare (0–99, în medie ~49) |
| `Complain` | 1 dacă s-a plâns în ultimii 2 ani (doar 21 de clienți) |

### 3. Cât a cheltuit (ultimii 2 ani) — coloanele `Mnt*` („Mnt” = amount)
| Coloană | Pe ce | Din totalul cheltuit de toți clienții |
|---|---|---|
| `MntWines` | vin | **50%** |
| `MntMeatProducts` | carne | **28%** |
| `MntGoldProds` | aur / produse premium | 7% |
| `MntFishProducts` | pește | 6% |
| `MntSweetProducts` | dulciuri | 4,5% |
| `MntFruits` | fructe | 4% |

Adică **vinul și carnea fac aproape 80% din bani**. Totalul pe client e între 5 și 2525, cu mediana **396** și media **606**. Media e mai mare decât mediana pentru că puțini clienți cheltuie foarte mult.

### 4. Cum cumpără — coloanele `Num*`
| Coloană | Ce înseamnă | În medie |
|---|---|---|
| `NumStorePurchases` | cumpărări în magazin | 5,8 |
| `NumWebPurchases` | cumpărări pe site | 4,1 |
| `NumCatalogPurchases` | cumpărări din catalog | 2,7 |
| `NumDealsPurchases` | cumpărări cu reducere (se suprapun cu cele de mai sus, nu sunt un canal separat) | 2,3 |
| `NumWebVisitsMonth` | vizite pe site în ultima lună | 5,3 |

### 5. Campaniile de marketing
Magazinul a trimis **6 campanii** de oferte, iar fiecare coloană spune dacă clientul a acceptat (1) sau nu (0):

| Coloană | Campania | Au acceptat |
|---|---|---|
| `AcceptedCmp1` | a 1-a | 6,4% |
| `AcceptedCmp2` | a 2-a | 1,3% (aproape nimeni) |
| `AcceptedCmp3` | a 3-a | 7,3% |
| `AcceptedCmp4` | a 4-a | 7,5% |
| `AcceptedCmp5` | a 5-a | 7,3% |
| **`Response`** | **ultima campanie** | **14,9%** |

`Response` e cea mai importantă coloană. E **ce vrem să prezicem**.

### 6. Gunoi
`Z_CostContact` (mereu 3) și `Z_Revenue` (mereu 11) au aceeași valoare la toată lumea. Probabil erau costul unui contact și venitul dintr-o acceptare, folosite de cine a creat setul. Nu spun nimic despre clienți.

## Cele două întrebări ale laboratorului

1. **„Cine va accepta oferta?”** → se prezice `Response` (da/nu) = **clasificare**.
2. **„Cât cheltuie un client în 2 ani?”** → se prezice suma celor 6 coloane `Mnt*` = **regresie**.

Ideea din spate: marketingul vrea să-și cheltuie bugetul pe clienții care **chiar răspund** și care **aduc bani**.

## Lucruri care sar în ochi (intuiții, nu concluzii)

- **Cine a mai acceptat o ofertă o acceptă mai ușor și pe următoarea.** Dintre clienții care au acceptat măcar una din campaniile 1–5, **41%** au acceptat-o și pe ultima. Dintre ceilalți, doar **8%**. Probabil va fi un predictor puternic.
- **Venitul și cheltuiala merg împreună** (corelație ≈ 0,67): cine câștigă mai mult cheltuie mai mult. Nu e o surpriză, dar e cel mai bun indiciu pentru regresie.
- **Majoritatea clienților (71%) au copii sau adolescenți acasă.**
- **Clienții sunt în mare parte de vârstă mijlocie:** anul median de naștere e 1970, deci ~44 de ani în 2014.

## Capcane de ținut minte

- **Nu folosi coloanele `Mnt*` ca să prezici cheltuiala.** Ele *sunt* cheltuiala, deci ar fi trișat (*leakage*).
- **Ținta `Response` e dezechilibrată (85% „nu”).** Un model care zice mereu „nu” are 85% acuratețe și e inutil. De aceea contează F1 și ROC-AUC.
- **Există clienți dublați:** 182 de rânduri sunt identice cu altele, doar `ID`-ul diferă.
- **Valori false:** venit 666666, ani de naștere 1893/1899/1900, stări civile „Absurd” și „YOLO”.
- **Date care nu se potrivesc:** 6 clienți au 0 cumpărări, dar au cheltuit bani.
- **`Dt_Customer` e text**, nu dată. Trebuie convertită înainte de orice calcul.

Detaliile tehnice sunt în [`lab2/lab2.md`](lab2/lab2.md).
