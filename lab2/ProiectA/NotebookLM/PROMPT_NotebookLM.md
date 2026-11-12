# Cum folosești fișierele în NotebookLM

## 1. Ce încarci ca surse (Add sources → Upload)

| Fișier | Ce conține |
|---|---|
| `ProiectA_Rezultate_Cod_Explicatii.pdf` | **tot** (ghid, proiect pas cu pas cu grafice, rezultate, Q&A, cod) – sursa principală |
| `01_Ghid_ML_pentru_incepatori.md` | noțiunile de ML explicate simplu, cu analogii |
| `02_Proiectul_pas_cu_pas.md` | fiecare pas: ce fac, de ce, de ce nu altceva + cifrele |
| `03_Rezultate_si_concluzii.md` | cele 7 lucruri esențiale, concluziile, limitările |
| `04_Intrebari_si_raspunsuri.md` | întrebări probabile la prezentare |
| `05_Codul_proiectului.md` | codul complet, pe secțiuni |

Ajunge și doar PDF-ul. Fișierele `.md` ajută NotebookLM să găsească mai ușor textul exact.

---

## 2. Promptul principal (îl lipești în chat)

```
Ești profesorul meu de Machine Learning. Explică-mi în română, ca unui începător total, tot Proiectul A din surse, de la date până la concluzii. Folosește doar informațiile și cifrele de acolo. Fiecare termen tehnic îl explici într-o propoziție, cu o analogie.

1. Povestea proiectului în 5 propoziții.
2. Pașii în ordine: ce am făcut, de ce și ce cifră a rezultat.
3. Ideile cheie: de ce acuratețea înșală (doar 15% răspund), class_weight și recall, data leakage, de ce Regresia Logistică, ce atribute contează, cât de sigure sunt rezultatele.
4. Cum citesc matricea de confuzie, curba ROC și importanța atributelor.
5. Un rezumat de 1 minut pentru prezentare și 10 întrebări probabile, cu răspunsuri.

Dacă ceva nu e în surse, spune asta în loc să inventezi.
```

---

## 3. Prompturi suplimentare (opționale)

**Pentru Audio Overview (Customize):**
```
Fă o discuție în română pentru un student care nu a lucrat niciodată cu Machine Learning. Explicați proiectul ca pe o poveste: magazinul vrea să știe cui să trimită oferta. Insistați pe: dezechilibrul claselor și de ce acuratețea minte, data leakage, alegerea Regresiei Logistice, ce atribute contează și cât de sigure sunt rezultatele. Folosiți analogii și cifrele exacte din surse (ROC-AUC 0,89, recall 0,76, R² 0,88).
```

**Pentru verificare (quiz):**
```
Pune-mi 10 întrebări, una câte una, despre Proiectul A, de la ușor la greu. Așteaptă răspunsul meu, corectează-mă și explică ce am greșit folosind sursele.
```

**Pentru cod:**
```
Explică-mi codul din secțiunea „Modelele și alegerea hiperparametrilor” linie cu linie, ca unui începător în Python: ce face Pipeline, GridSearchCV și class_weight="balanced".
```

**Pentru prezentare:**
```
Fă-mi un plan de prezentare de 7 minute pentru Proiectul A, cu ce spun pe fiecare slide și ce grafic arăt, folosind doar sursele.
```
