# Întrebări probabile (de la profesor / colegi) și răspunsuri scurte

**1. De ce nu folosești acuratețea?**
Pentru că doar 15% dintre clienți răspund. Un model care spune mereu „nu” are 85% acuratețe și nu găsește niciun client. În proiect, Dummy-ul are 75% acuratețe și ROC-AUC 0,5, iar Random Forest are cea mai mare acuratețe (87,5%) deși ratează jumătate din clienții interesați.

**2. Ce înseamnă ROC-AUC 0,89?**
Dacă iau la întâmplare un client care a răspuns și unul care n-a răspuns, în 89% din cazuri modelul îi dă o probabilitate mai mare celui care a răspuns. 0,5 = ghicit la întâmplare, 1 = perfect.

**3. Ce este data leakage și cum l-ai evitat?**
Informație din test care ajunge în antrenare și face scorurile prea optimiste. L-am evitat împărțind datele la început, punând preprocesarea (imputare, scalare, SMOTE) în pipeline și scoțând coloanele `Mnt*` la regresie. Dovada: scorul din CV s-a confirmat pe test.

**4. De ce `class_weight` și nu SMOTE?**
Au dat aproape aceleași rezultate (recall 0,805 vs. 0,798), dar `class_weight` e mai simplu: nu creează clienți artificiali și nu adaugă un pas în plus.

**5. De ce ai ales Regresia Logistică dacă SVM a avut același scor?**
La scoruri egale aleg modelul mai simplu și explicabil. Pot arăta unui manager exact ce atribute cresc sau scad șansa de răspuns.

**6. De ce scalezi datele?**
SVM și Regresia Logistică depind de mărimea numerelor. Venitul (~50.000) ar domina numărul de copii (0–2). Experimentul arată: SVM fără scalare scade de la 0,90 la 0,62 ROC-AUC.

**7. De ce mediana și nu media pentru venitul lipsă?**
Venitul e asimetric (câțiva clienți foarte bogați); media ar fi trasă în sus, mediana nu.

**8. De ce One-Hot și nu 0, 1, 2, 3?**
Numerotarea ar inventa o ordine falsă (ex. „Married < Single”). One-Hot face câte o coloană 0/1 pentru fiecare categorie.

**9. De ce 5-fold cross-validation?**
Cu doar ~330 de clienți pozitivi în train, un singur set de validare ar da un scor la noroc. Media pe 5 fold-uri e mai stabilă și văd și deviația standard. 10 fold-uri ar fi durat de două ori mai mult.

**10. Are vreun model overfitting?**
Random Forest: 1,0 pe train, ~0,89 pe validare. Regresia Logistică nu: train 0,919 vs. validare 0,904.

**11. Cât de sigure sunt rezultatele?**
Pe 10 împărțiri diferite, ROC-AUC = 0,892 ± 0,010. Diferențele dintre modele sunt mai mici decât această variație.

**12. Ce sunt clienții duplicați și contează?**
74 de rânduri din test apar identic și în train. Fără ele scorul nu scade (0,890 → 0,909), deci nu au umflat rezultatul.

**13. Ce atribute contează cel mai mult?**
`Recency` (cât de recent a cumpărat), vechimea clientului, cheltuielile pe carne, starea civilă, cumpărăturile din catalog, campaniile acceptate anterior.

**14. De ce Random Forest câștigă la regresie dar nu la clasificare?**
La clasificare relația e aproape liniară, deci un model liniar e suficient. La cheltuieli relația nu e liniară (Ridge R² 0,80 vs. Random Forest 0,88), iar arborii prind asta.

**15. Ce ai face diferit?**
Aș alege pragul de decizie după profit (costul unui contact vs. câștigul unui client), aș încerca gradient boosting și aș transforma logaritmic ținta de regresie.
