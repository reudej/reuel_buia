#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define HRANA_SACHOVNICE 8

int hrana_sachovnice;

// -------------------------------------------------------------
// Pomocné funkce
// -------------------------------------------------------------

bool is_valid(const int *queens, int len) {
    for (int i = 0; i < len; i++) {
        for (int j = i + 1; j < len; j++) {
            if (queens[i] == queens[j] ||
                abs(queens[i] - queens[j]) == abs(i - j))
                return false;
        }
    }
    return true;
}

void print_solution(const int *queens) {
    printf("[");
    for (int i = 0; i < hrana_sachovnice; i++) {
        printf("%d", queens[i]);
        if (i < hrana_sachovnice - 1) printf(", ");
    }
    printf("]\n");
}

// -------------------------------------------------------------
// 1) Brute force – zkouší všechny permutace
// -------------------------------------------------------------

bool next_permutation(int *arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] > arr[i + 1]) i--;
    if (i < 0) return false;
    int j = n - 1;
    while (arr[j] < arr[i]) j--;
    int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    for (int a = i + 1, b = n - 1; a < b; a++, b--) {
        tmp = arr[a]; arr[a] = arr[b]; arr[b] = tmp;
    }
    return true;
}

bool brute_force(int *result) {
    int perm[hrana_sachovnice];
    for (int i = 0; i < hrana_sachovnice; i++) perm[i] = i;

    do {
        if (is_valid(perm, hrana_sachovnice)) {
            for (int i = 0; i < hrana_sachovnice; i++) result[i] = perm[i];
            return true;
        }
    } while (next_permutation(perm, hrana_sachovnice));

    return false;
}

// -------------------------------------------------------------
// 2) DFS / backtracking
// -------------------------------------------------------------

bool dfs_backtrack(int row, int *queens) {
    if (row == hrana_sachovnice) return true;

    for (int col = 0; col < hrana_sachovnice; col++) {
        bool safe = true;
        for (int r = 0; r < row; r++) {
            int c = queens[r];
            if (c == col || abs(row - r) == abs(col - c)) {
                safe = false;
                break;
            }
        }
        if (safe) {
            queens[row] = col;
            if (dfs_backtrack(row + 1, queens)) return true;
        }
    }
    return false;
}

// -------------------------------------------------------------
// 3) Náhodné hledání
// -------------------------------------------------------------

bool random_search(int *result, int max_tries) {
    for (int t = 0; t < max_tries; t++) {
        for (int i = 0; i < hrana_sachovnice; i++)
            result[i] = rand() % hrana_sachovnice;
        if (is_valid(result, hrana_sachovnice))
            return true;
    }
    return false;
}

// -------------------------------------------------------------
// 4) Postupné umisťování (heuristika)
// -------------------------------------------------------------

bool incremental_safe_placement(int *result, int max_restarts) {
    for (int attempt = 0; attempt < max_restarts; attempt++) {
        bool col_used[hrana_sachovnice] = {0};
        bool diag1[2*hrana_sachovnice] = {0}; // r - c + N
        bool diag2[2*hrana_sachovnice] = {0}; // r + c
        bool success = true;

        for (int row = 0; row < hrana_sachovnice; row++) {
            int safe_cols[hrana_sachovnice];
            int safe_count = 0;

            for (int col = 0; col < hrana_sachovnice; col++) {
                if (!col_used[col] &&
                    !diag1[row - col + hrana_sachovnice] &&
                    !diag2[row + col]) {
                    safe_cols[safe_count++] = col;
                }
            }

            if (safe_count == 0) {
                success = false;
                break;
            }

            int chosen = safe_cols[rand() % safe_count];
            result[row] = chosen;
            col_used[chosen] = true;
            diag1[row - chosen + hrana_sachovnice] = true;
            diag2[row + chosen] = true;
        }

        if (success && is_valid(result, hrana_sachovnice))
            return true;
    }
    return false;
}

// -------------------------------------------------------------
// Pomocná funkce pro měření času
// -------------------------------------------------------------

void measure(bool (*func)(int *), const char *name) {
    int queens[hrana_sachovnice];
    clock_t start = clock();
    bool found = func(queens);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%-30s trvalo %.6f s\n", name, elapsed);
    if (found) {
        printf("  Nalezeno řešení: ");
        print_solution(queens);
    } else {
        printf("  Nebylo nalezeno řešení.\n");
    }
    printf("\n");
}

// -------------------------------------------------------------
// Hlavní funkce
// -------------------------------------------------------------

int main(int argc, char **argv) {

    if (argc > 2) {
        fprintf(stderr, "Použití: %s [<hrana_sachovnice>]", )
    }

    printf("Porovnání 4 algoritmů pro problém 8 královen:\n\n");

    measure(brute_force, "1) Brute force");
    measure(dfs_backtrack, "2) DFS/backtracking");
    measure(
        (bool (*)(int *)) (^(int *q){ return random_search(q, 1000000); }),
        "3) Náhodné hledání"
    );
    measure(
        (bool (*)(int *)) (^(int *q){ return incremental_safe_placement(q, 10000); }),
        "4) Postupné umisťování"
    );

    return 0;
}
