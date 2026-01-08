#!/usr/bin/python3

# Importy potřebné jak pro generátor tak pro prediktor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

pd.set_option("future.no_silent_downcasting", True)


# Importy potřebné jen pro generátor
from traceback import print_exc
from sys import stdout, stderr, argv
from subprocess import run
from os.path import exists
import time

if len(argv) == 0:
    print("Použití: <Tento program> <Cesta/název souboru s daty ve formátu csv> <Cesta/název výsledného prediktoru>", file=stderr)
    exit(2)
if len(argv) != 3:
    print("Použití:", argv[0], "<Cesta/název souboru s daty ve formátu csv> <Cesta/název výsledného prediktoru>", file=stderr)
    exit(1)

SABLONA = "sablona"
NAZEV_USEKU_KODU = ".usek_kodu_pro_upravu_dat_pred_predikci.py"

data = pd.read_csv(argv[1])

while True:
    while True:
        try:
            odpoved = input("Chcete do kódu před samotnou predikci přidat úpravu dat (A/N)? ").upper().strip()
        except KeyboardInterrupt:
            print()
            exit()
        except Exception:
            print("Chybný vstup vyvolal výjimku:\n")
            print_exc()
        else:
            if odpoved == "A" or odpoved == "N":
                break

    if odpoved == "A":
        run(["nano", NAZEV_USEKU_KODU])
        if not exists(NAZEV_USEKU_KODU):
            continue
    break

vlozeny_usek = ""
if odpoved =="A":
    with open(NAZEV_USEKU_KODU) as f:
        vlozeny_usek = f.read()
        exec(vlozeny_usek)

print("Zadejte prosím čísla sloupců, které mají být predikovány, oddělujte mezerou:\n")
print("Pro ukončení programu stiskněte Ctrl+C.\n")
print("Možnosti:")
for i, nazev in enumerate(data.columns):
    print(f"    {i}) {nazev}")
print()

X_all_cols = set(data.columns)
y_all_cols = set()

while True:
    try:
        y_all_cols.update(set([list(data.columns)[int(i)] for i in input("Čísla sloupců: ").split(" ")]))
    except KeyboardInterrupt:
        print()
        exit()
    except Exception:
        print("Chybný vstup vyvolal výjimku:\n")
        print_exc()
    else:
        break

while True:
    try:
        inp = input("Zadejte prosím čísla sloupců, ktreré chcete vyloučit ze vstupu: ").split(" ")
        if inp == ['']: break
        X_all_cols.difference_update(set([list(data.columns)[int(i)] for i in inp]))
    except KeyboardInterrupt:
        print()
        exit()
    except Exception:
        print("Chybný vstup vyvolal výjimku:\n")
        print_exc()
    else:
        break


while True:
    try:
        min_pocet_stromu = int(input("Minimální počet stromů: "))
    except KeyboardInterrupt:
        print()
        exit()
    except Exception:
        print("Chybný vstup vyvolal výjimku:\n")
        print_exc()
    else:
        break
while True:
    try:
        max_pocet_stromu = int(input("Maximální počet stromů: "))
    except KeyboardInterrupt:
        print()
        exit()
    except Exception:
        print("Chybný vstup vyvolal výjimku:\n")
        print_exc()
    else:
        break
prum_pocet_stromu = (min_pocet_stromu + max_pocet_stromu) //2

print("Jaký zlomek z celkové části dat má být použit k testování natrénovaného modelu?")
while True:
    try:
        test_size_frac = float(input("Zadejte ho sem jako desetinné číslo: ").replace(",", "."))
    except KeyboardInterrupt:
        print()
        exit()
    except Exception:
        print("Chybný vstup vyvolal výjimku:\n")
        print_exc()
    else:
        break

X_all_cols.difference_update(y_all_cols)

X_all_cols, y_all_cols = list(X_all_cols), list(y_all_cols)
if len(X_all_cols) == 1: X_all_cols = X_all_cols[0]
if len(y_all_cols) == 1: y_all_cols = y_all_cols[0]
X_all, y_all = data[X_all_cols], data[y_all_cols]
"""
print(X_all.head(), y_all.head(), sep="\n")
exit()
"""

X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=test_size_frac)

test_size = len(X_test)


print()

logis_regr = LogisticRegression()

print("Probíhá trénování logisticky regresního modelu...", end="")
stdout.flush()

start = time.perf_counter()
logis_regr.fit(X_train, y_train)
end = time.perf_counter()
lin_regr_trenovaci_cas = end - start

print(" Natrénováno. Trénovací čas:", lin_regr_trenovaci_cas, "s")

print("Probíhá test natrénovaného modelu...")
stdout.flush()

start = time.perf_counter()
logis_regr_y_pred_acc = logis_regr.predict(X_test)
logis_regr_y_pred_loss = logis_regr.predict_proba(X_test)
end = time.perf_counter()
logis_regr_prum_pred_cas = (end-start) / test_size

print(" Test dokončen. Průměrný predikční čas:", logis_regr_prum_pred_cas, "s")

logis_regr_acc = accuracy_score(y_test, logis_regr_y_pred_acc)
logis_regr_loss = log_loss(y_test, logis_regr_y_pred_loss)
print("ACCURACY:", logis_regr_acc)
print("LOSS:", logis_regr_loss)
print()



print("Probíhá automatické nastavování počtu stromů u náhodného lesa...")

zanadbatelny_rozdil_acc = 0.005
vychozi_dict = {"pripraveno": False}
rand_forests = [rand_forest.copy() for rand_forest in [vychozi_dict]*3]
while True:
    if min_pocet_stromu == max_pocet_stromu:
        rand_forest = rand_forest_min
        pocet_stromu = min_pocet_stromu
        break

    rand_forest_min, rand_forest_max, rand_forest_avg = rand_forests

    prum_pocet_stromu = (min_pocet_stromu + max_pocet_stromu) //2
    if prum_pocet_stromu == min_pocet_stromu or prum_pocet_stromu == max_pocet_stromu:
        prum_pocet_stromu = -1
        rand_forest_avg["pripraveno"] = True
        rand_forest_avg["acc"] = -1
    for rand_forest, pocet_stromu in zip(rand_forests, (min_pocet_stromu, max_pocet_stromu, prum_pocet_stromu)):
        if not rand_forest["pripraveno"] :
            rand_forest["model"] = RandomForestClassifier(n_estimators=pocet_stromu)

    for rand_forest in rand_forests:
        if rand_forest["pripraveno"]:
            continue

        start = time.perf_counter()
        rand_forest["model"].fit(X_train, y_train)
        end = time.perf_counter()
        rand_forest["trenovaci_cas"] = end-start

        start = time.perf_counter()
        rand_forest["y_pred_acc"] = rand_forest["model"].predict(X_test)
        rand_forest["y_pred_loss"] = rand_forest["model"].predict_proba(X_test)
        end = time.perf_counter()
        rand_forest["prum_pred_cas"] = (end-start) / test_size

        rand_forest["pripraveno"] = True

        rand_forest["acc"] = accuracy_score(y_test, rand_forest["y_pred_acc"])
        rand_forest["loss"] = log_loss(y_test, rand_forest["y_pred_loss"])

    acc_min, acc_max, acc_avg = [rand_forest["acc"] for rand_forest in rand_forests]

    if abs(acc_min-acc_max) <= zanadbatelny_rozdil_acc:
        rand_forest = rand_forest_min
        pocet_stromu = min_pocet_stromu
        break
    if acc_avg == -1:
        rand_forest = rand_forest_max
        pocet_stromu = max_pocet_stromu
        break
    if abs(acc_avg-acc_max) <= zanadbatelny_rozdil_acc:
        min_pocet_stromu = prum_pocet_stromu
        rand_forests[:] = [rand_forest_min, rand_forest_avg, vychozi_dict.copy()]
        continue

    max_pocet_stromu = prum_pocet_stromu
    rand_forests[:] = [rand_forest_avg, rand_forest_max, vychozi_dict.copy()]

print(" Nastaveno na:", pocet_stromu)
print("\nInformace o výsledcích náhodného lesa:")
print("Trénovací čas:", rand_forest["trenovaci_cas"])
print("Průměrný predikční čas:", rand_forest["prum_pred_cas"])
print("ACCURACY:", rand_forest["acc"])
print("LOSS:", rand_forest["loss"])
print()

modely = (("LogisticRegression()", "from sklearn.linear_model import LinearRegression"),
          (f"RandomForestClassifier(n_estimators={pocet_stromu})", "from sklearn.ensemble import RandomForestClassifier"))
"""
if rand_forest["rmse"] < logis_regr_loss:
    print("Bude použit náhodný les.")
    model = 1
else:
    print("V tomto případě bude lepší použít logistickou regresi.")
    model = 0
"""
model = 1
model, model_imp = modely[model]

print("Generuje se výsledný soubor...")


with open(SABLONA, encoding="utf-8") as f:
    sablona = f.read()

with open(argv[2], mode="w", encoding="utf-8") as f:
    f.write(sablona.format(
        csv_soubor=argv[1],
        X_all_cols=repr(X_all_cols),
        y_all_cols=repr(y_all_cols),
        vlozeny_usek=vlozeny_usek,
        test_size_frac=test_size_frac,
        model_imp=model_imp,
        model=model
    ))

print(" Hotovo.")
