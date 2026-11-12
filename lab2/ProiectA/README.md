# Proiect A — Customer Personality Analysis

| Fișier | Ce este |
|---|---|
| `ProiectA_Customer_Personality.ipynb` | notebook-ul **cu specificații** (comentarii + observații) — se predă pe Classroom |
| `ProiectA_Customer_Personality_prezentare.ipynb` | același cod, **fără specificații** (doar titlurile pașilor) — pe el prezinți |
| `data/raw.csv` | setul de date (dacă lipsește, notebook-ul îl descarcă de pe Kaggle cu `kagglehub`) |
| `data/clean.csv`, `figures/`, `results/`, `models/` | create de notebook la rulare |

## Rulare locală

Din folderul `ML` (unde e `.venv`), o singură dată:

```bash
.venv\Scripts\python.exe -m pip install -r lab2\ProiectA\requirements.txt
```

Apoi deschizi notebook-ul în VS Code / Jupyter, alegi kernelul din `.venv` și dai **Run All**.
Sau din linia de comandă (din `lab2\ProiectA`):

```bash
..\..\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 ProiectA_Customer_Personality.ipynb
```

Rularea completă durează aproximativ 20–30 de minute pe CPU. Cel mai lent pas este capitolul 10 (rețelele Keras).
Rezultatele sunt reproductibile (`SEED = 42` peste tot). Excepție: rețelele pot varia la a treia zecimală, de aceea capitolul 10.3 raportează media pe 5 seed-uri.

## Ordinea capitolelor (= baremul)

0 setări și funcții · 1 încărcare · 2 EDA · 3 curățare · 4 atribute + design train/test · 5 preprocesare în pipeline + leakage ·
6 dezechilibru · 7 modele + evaluare inițială CV + curbe ROC/PR · 8 grid search + senzitivitate + nested CV ·
9 ansambluri + test t corectat · 10 rețele neuronale · 11 evaluare finală pe test + profit · 12 concluzii
