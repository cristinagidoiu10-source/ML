# Ghid de Machine Learning pentru începători (pe exemplul Proiectului A)

> Acest document explică, fără să presupună cunoștințe anterioare, toate noțiunile de Machine Learning (ML) folosite în proiect. Fiecare noțiune are: **ce este**, **o analogie din viața reală** și **unde apare în proiect**.

---

## 1. Ce este Machine Learning?

**Machine Learning** = calculatorul învață reguli din exemple, în loc ca un om să scrie regulile de mână.

- **Analogie:** un vânzător cu experiență „simte” care clienți vor cumpăra o ofertă, pentru că a văzut sute de clienți înainte. Un model ML face același lucru, dar cu mii de clienți și cu calcule matematice.
- **În proiect:** calculatorul primește date despre 2240 de clienți ai unui magazin (venit, vârstă, copii, ce cumpără) și **învață să ghicească** dacă un client va accepta o ofertă de marketing.

## 2. Învățare supervizată

Avem exemple pentru care **știm deja răspunsul corect** (eticheta). Modelul învață legătura dintre date și răspuns, apoi o aplică pe clienți noi.

- **Atribute (features, X)** = informațiile despre client (venit, vârstă, cheltuieli…). Sunt „întrebările” pe care modelul le poate pune.
- **Țintă (target, y)** = ce vrem să prezicem. Aici: `Response` (1 = a acceptat oferta, 0 = nu).

## 3. Clasificare vs. regresie

| | Clasificare | Regresie |
|---|---|---|
| Ce prezice | o **categorie** (da/nu) | un **număr** |
| În proiect | „Va accepta clientul oferta?” | „Câți bani va cheltui clientul?” |
| Model ales | Regresia Logistică | Random Forest Regressor |

Proiectul răspunde la **ambele** întrebări; accentul principal e pe clasificare.

## 4. Analiza exploratorie a datelor (EDA)

Înainte de orice model ne **uităm la date**: ce coloane sunt, ce lipsește, ce valori sunt ciudate, cum arată ținta.

- **Analogie:** înainte să gătești, verifici ce ingrediente ai în frigider și dacă vreunul e stricat.
- **În proiect:** am găsit clienți „născuți” în 1893 (vârstă > 120 de ani) și un venit de 666.666 – erori evidente, eliminate. Am văzut că doar 15% dintre clienți au răspuns.

## 5. Clase dezechilibrate (cea mai importantă capcană a proiectului!)

Doar **15%** dintre clienți au acceptat oferta, 85% nu. Asta se numește **dezechilibru între clase**.

- **Problema:** un model „leneș” care spune mereu „NU” are **85% acuratețe**, dar e complet inutil – nu găsește niciun client interesat.
- **Analogie:** un detector de incendiu care nu sună niciodată „are dreptate” 99,9% din zile, dar e inutil exact când contează.
- **Soluția din proiect:** `class_weight="balanced"` – îi spunem modelului că o greșeală pe un client care ar fi răspuns costă de ~6 ori mai mult. Alternativa testată: **SMOTE** (creează clienți „sintetici” din clasa rară). Au dat rezultate similare; s-a ales `class_weight` pentru că e mai simplu.

## 6. Împărțirea train / test

Datele se împart în:
- **Train (80%)** – modelul învață din ele;
- **Test (20%)** – puse deoparte, folosite **o singură dată, la final**, ca să vedem cât de bine merge modelul pe clienți pe care nu i-a văzut niciodată.

- **Analogie:** elevul învață din culegere (train), dar nota se dă pe un examen cu probleme noi (test). Dacă examenul ar avea exact problemele din culegere, nota n-ar arăta nimic.
- **Stratificat:** ambele părți păstrează aceeași proporție de 15% clienți care răspund.

## 7. Validarea încrucișată (cross-validation, k-fold)

Ca să alegem modelul fără să atingem testul, împărțim train-ul în **5 bucăți**. De 5 ori: antrenăm pe 4 bucăți și verificăm pe a 5-a. Facem media celor 5 scoruri.

- **Analogie:** în loc de un singur test de verificare (poate norocos), dai 5 teste diferite și faci media.
- **De ce:** cu doar ~330 de clienți care răspund în train, un singur test ar fi prea „la noroc”. Media pe 5 e stabilă, iar **deviația standard** arată cât variază scorul.

## 8. Preprocesarea și pipeline-ul

Modelele lucrează doar cu **numere complete**. Preprocesarea pregătește datele:

| Pas | Ce face | De ce |
|---|---|---|
| **Imputare (mediană)** | completează cele 24 de venituri lipsă cu mediana | mediana nu e influențată de valori extreme (spre deosebire de medie) |
| **Scalare (StandardScaler)** | aduce toate numerele la aceeași scară (medie 0, deviație 1) | altfel venitul (~50.000) „ar strivi” numărul de copii (0–2) |
| **One-Hot Encoding** | transformă textul („Married”, „PhD”) în coloane 0/1 | modelele nu înțeleg text; nu folosim 1,2,3 ca să nu inventăm o ordine falsă |

**Pipeline** = o „bandă de producție” care leagă toți pașii de model. Orice client trece automat prin aceiași pași, în aceeași ordine.

## 9. Data leakage („scurgerea” de informație) – a doua mare capcană

**Data leakage** = informație din test ajunge, direct sau indirect, în antrenare → scoruri prea frumoase, care nu se confirmă în realitate.

- **Analogie:** elevul vede subiectele examenului înainte. Nota iese mare, dar nu înseamnă că știe materia.
- **Cum am evitat-o în proiect:**
  1. împărțirea train/test se face **înainte** de orice calcul care învață din date (medii, mediane);
  2. preprocesarea stă **în pipeline**, deci în fiecare fold de cross-validation se calculează doar pe partea de antrenare;
  3. SMOTE e pus în pipeline (`imblearn`), ca exemplele sintetice să nu ajungă în validare;
  4. la regresie am scos coloanele `Mnt*` din atribute (ținta e chiar suma lor – altfel modelul ar „copia” răspunsul);
  5. am verificat clienții duplicați dintre train și test.

## 10. Modelele folosite

| Model | Explicație simplă | Analogie |
|---|---|---|
| **Dummy (baseline)** | ghicește la întâmplare, după proporții | aruncarea unei monede – orice model serios trebuie să-l bată |
| **Regresia Logistică** | dă fiecărui atribut o „greutate” (pozitivă sau negativă), le adună și transformă suma într-o probabilitate | un formular de punctaj: +puncte dacă a cumpărat recent, −puncte dacă are copii… |
| **SVM (Support Vector Machine)** | caută granița care separă cel mai bine cele două grupuri, cu cea mai mare „distanță de siguranță” | trasezi o șosea cât mai lată între două sate |
| **Random Forest** | sute de **arbori de decizie** (serii de întrebări da/nu), fiecare antrenat puțin diferit; votează împreună | consulți 300 de experți și iei decizia majorității |

## 11. Hiperparametri și Grid Search

**Hiperparametrii** = „butoanele de reglaj” ale unui model, pe care le alegem noi înainte de antrenare (ex. `C` la Regresia Logistică = cât de simplu forțăm modelul să fie).

**Grid Search** = încercăm **toate combinațiile** dintr-o listă de valori, fiecare evaluată cu 5-fold CV, și o păstrăm pe cea mai bună.

- **Analogie:** reglezi temperatura și timpul la cuptor – încerci mai multe combinații și o păstrezi pe cea cu cel mai bun cozonac.

## 12. Overfitting și underfitting

- **Overfitting (supraînvățare):** modelul „memorează” datele de antrenare, inclusiv zgomotul → scor perfect pe train, slab pe date noi. *Elevul care învață pe de rost răspunsurile, fără să înțeleagă.*
- **Underfitting:** modelul e prea simplu și nu prinde nici tiparele reale.
- **Cum se vede:** comparăm scorul pe train cu scorul pe validare. Diferență mare = overfitting. În proiect, Random Forest ajunge la 1.0 pe train, dar ~0.89 pe validare.

## 13. Metricile de evaluare (cum dăm nota modelului)

Mai întâi **matricea de confuzie** – un tabel 2×2 cu toate tipurile de răspunsuri:

| | Model: „nu răspunde” | Model: „răspunde” |
|---|---|---|
| **Realitate: nu răspunde** | corect (adevărat negativ) | **ofertă trimisă degeaba** (fals pozitiv) |
| **Realitate: răspunde** | **client ratat** (fals negativ) | corect (adevărat pozitiv) |

| Metrică | Întrebarea la care răspunde | În proiect |
|---|---|---|
| **Acuratețe** | Câte predicții sunt corecte în total? | **Înșelătoare** aici (85% doar spunând mereu „nu”) |
| **Recall** | Din clienții care ar răspunde, pe câți îi găsim? | **cea mai importantă pentru business**: ~0,76–0,79 |
| **Precizie** | Din cei cărora le trimitem oferta, câți răspund? | ~0,44 |
| **F1** | Echilibrul dintre precizie și recall | ~0,56 |
| **ROC-AUC** | Cât de bine *ordonează* modelul clienții de la „cel mai probabil” la „cel mai puțin probabil”? 1 = perfect, 0,5 = la întâmplare | **0,89** pe test |
| **PR-AUC** | ca ROC-AUC, dar concentrată pe clasa rară; „la întâmplare” = 0,15 | ~0,61 |

**De ce recall înainte de precizie?** A rata un client interesat (o vânzare pierdută) costă mai mult decât un email trimis degeaba.

**Pentru regresie:**
- **MAE** = eroarea medie, în bani (ex. „greșim în medie cu 108 unități”);
- **RMSE** = similar, dar penalizează mai tare greșelile mari;
- **R²** = cât din diferențele dintre clienți explică modelul (1 = tot, 0 = nimic). Aici 0,88.

## 14. Permutation importance (ce atribute contează)

Amestecăm la întâmplare valorile unei singure coloane și vedem cât scade scorul. Dacă scade mult, modelul se baza pe acea coloană.

- **Analogie:** ca să vezi cât contează busola pentru un drumeț, i-o strici și vezi cât de mult se rătăcește.

## 15. Studiu de ablație („cu / fără”)

Scoatem pe rând câte o componentă (scalarea, `class_weight`, atributele noi) și vedem cât de mult se strică rezultatul. Așa **demonstrăm** că fiecare decizie a contat, nu doar o afirmăm.

## 16. Reproductibilitate (`SEED = 42`)

Multe operații folosesc numere aleatoare (împărțirea datelor, Random Forest). Fixând „sămânța” (`SEED = 42`), rulând din nou obținem **exact aceleași rezultate**.
