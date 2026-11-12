"""Lab 2 — T8 Customer Personality Analysis.

Un lant de magazine pregateste o noua campanie de oferte. Marketingul vrea sa stie:
  1. ce clienti vor accepta oferta      -> clasificare (tinta: Response)
  2. cat cheltuie un client in doi ani  -> regresie   (tinta: suma coloanelor Mnt*)

Fiecare TODO are fisierul lui (todo1.py, todo2.py, ...) si se poate rula separat:
    python lab2\\todo1.py

Acest fisier ruleaza toate TODO-urile, in ordine:
    python lab2\\lab2.py
"""

import glob
import os
import re
import runpy

LAB_DIR = os.path.dirname(os.path.abspath(__file__))


def todo_number(path):
    return int(re.search(r"todo(\d+)\.py$", path).group(1))


if __name__ == "__main__":
    for path in sorted(glob.glob(os.path.join(LAB_DIR, "todo*.py")), key=todo_number):
        print(f"\n{'=' * 30} {os.path.basename(path)} {'=' * 30}")
        runpy.run_path(path, run_name="__main__")
