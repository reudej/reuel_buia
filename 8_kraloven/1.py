import time
import itertools
import random

N = 8

# -----------------------------------------------------
# Pomocné funkce
# -----------------------------------------------------

def is_valid(queens):
    """Zkontroluje, zda žádné dvě královny se nenapadají."""
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            if queens[i] == queens[j] or abs(queens[i] - queens[j]) == abs(i - j):
                return False
    return True


# -----------------------------------------------------
# 1) Brute Force – všechny možné kombinace
# -----------------------------------------------------

def brute_force():
    solutions = []
    cols = range(N)
    for perm in itertools.permutations(cols):
        if is_valid(perm):
            solutions.append(perm)
    return solutions


# -----------------------------------------------------
# 2) DFS / Backtracking
# -----------------------------------------------------

def dfs_backtrack(row=0, queens=[]):
    if row == N:
        return [queens[:]]
    solutions = []
    for col in range(N):
        # kontrola bezpečnosti
        safe = True
        for r, c in enumerate(queens):
            if c == col or abs(row - r) == abs(col - c):
                safe = False
                break
        if safe:
            queens.append(col)
            solutions += dfs_backtrack(row + 1, queens)
            queens.pop()
    return solutions


# -----------------------------------------------------
# 3) Náhodné pokusy
# -----------------------------------------------------

def random_search_(reseni, max_tries=1_000_000):
    pocet_kroku = 0
    for _ in range(max_tries):
        pocet_kroku += 2*N +2
        reseni[:] = [random.randint(0, N - 1) for _ in range(N)]
        if is_valid(queens):
            return pocet_kroku
    reseni[:] = [None]
    return pocet_kroku
def random_search(times=10, max_tries=1_000_000):
    reseni = set()
    pocet_kroku = 0
    for _ in times:
        reseni_ = []


# -----------------------------------------------------
# 4) Postupné bezpečné umisťování
# -----------------------------------------------------
def incremental_safe_placement_(vylouceno_ob, proz_reseni, reseni):
    
def incremental_safe_placement():
    reseni = set()
    pocet_kroku = 0
    for row in range(N):
        pocet_kroku += 1
        for col in range(N):
            pole = [None]*N
            pole[row] = col
            pocet_kroku += 3
            pocet_kroku += incremental_safe_placement_(set(), pole, reseni)
    return pocet_kroku, reseni
    """
    max_restarts = 10000
    for _ in range(max_restarts):
        queens = []
        cols = set()
        diag1 = set()  # r - c
        diag2 = set()  # r + c

        success = True
        for row in range(N):
            safe_cols = [c for c in range(N)
                         if c not in cols and
                            (row - c) not in diag1 and
                            (row + c) not in diag2]
            if not safe_cols:
                success = False
                break
            col = random.choice(safe_cols)
            queens.append(col)
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

        if success and is_valid(queens):
            return queens
    return None
    """

"""
# -----------------------------------------------------
# Porovnání časů
# -----------------------------------------------------

def measure_time(func, name):
    start = time.time()
    result = func()
    end = time.time()
    print(f"{name:<30} trvalo {end - start:.4f} s")
    if isinstance(result, list) and len(result) > 0:
        print(f"  Nalezeno řešení: {result}")
    elif result:
        print(f"  Nalezeno řešení: {result}")
    else:
        print("  Nebylo nalezeno řešení.")
    print()

"""


if __name__ == "__main__":
    print("Porovnání čtyř algoritmů pro problém 8 královen:\n")

    measure_time(brute_force, "1) Brute force")
    measure_time(lambda: dfs_backtrack(), "2) DFS/backtracking")
    measure_time(random_search, "3) Náhodné hledání")
    measure_time(incremental_safe_placement, "4) Postupné umisťování")
