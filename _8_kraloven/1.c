#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define VYCHOZI_HRANA_SACHOVNICE 8
#define POCET_ALGORITMU 4
#define RANDOM_SEARCH_KOLIKRAT 10 //Počet náhodných hledání pro zprůměrování počtu kroků

signed long hrana_sachovnice;
unsigned long hrana_sachovnice_krat_3;

// -------------------------------------------------------------
// Pomocné funkce
// -------------------------------------------------------------

static inline bool is_valid(const signed long *queens) {
    for (signed long i = 0; i < hrana_sachovnice; i++) {
        for (signed long j = i + 1; j < hrana_sachovnice; j++) {
            if (queens[i] == queens[j] ||
                labs(queens[i] - queens[j]) == labs(i - j))
                return false;
        }
    }
    return true;
}

void print_solution(const signed long *queens) {
    printf("[");
    for (int i = 0; i < hrana_sachovnice; i++) {
        printf("%ld", queens[i]);
        if (i < hrana_sachovnice - 1) printf(", ");
    }
    printf("]\n");
}

// -------------------------------------------------------------
// 1) Brute force – zkouší všechny permutace
// -------------------------------------------------------------

static inline bool next_permutation(signed long *arr) {
    int i = hrana_sachovnice - 2;
    while (i >= 0 && arr[i] > arr[i + 1]) i--;
    if (i < 0) return false;
    int j = hrana_sachovnice - 1;
    while (arr[j] < arr[i]) j--;
    signed long tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    for (int a = i + 1, b = hrana_sachovnice - 1; a < b; a++, b--) {
        tmp = arr[a]; arr[a] = arr[b]; arr[b] = tmp;
    }
    return true;
}

bool brute_force(signed long *result, unsigned long *pocet_kroku) {
    for (int i = 0; i < hrana_sachovnice; i++) result[i] = i;

    do {
        if (is_valid(result)) return true;
    } while (next_permutation(result));

    return false;
}

// -------------------------------------------------------------
// 2) DFS / backtracking
// -------------------------------------------------------------

bool dfs_backtrack(signed long *queens, unsigned long *pocet_kroku) {
    signed long row = 0;
    signed long col = 0;
    while (1) {
        for (; col < hrana_sachovnice; col++) {
            *pocet_kroku += 3;
            bool safe = true;
            for (signed long r = 0; r < row; r++) {
                signed long c = queens[r];
                if (c == col || labs(row - r) == labs(col - c)) {
                    safe = false;
                    break;
                }
            }
            if (safe) {
                queens[row++] = col;
                if (row == hrana_sachovnice) return true;
                col = 0;
            }
        }
        row--;
        if (!row) return false;
        col = queens[row] + 1;
    }
}

// -------------------------------------------------------------
// 3) Náhodné hledání
// -------------------------------------------------------------

bool random_search(signed long *result, unsigned long *pocet_kroku) {
    while (1) {
        for (int i = 0; i < hrana_sachovnice;i++) result[i] = rand() % hrana_sachovnice;
        if (is_valid(result))
            return true;
    }
    return false;
}

// -------------------------------------------------------------
// 4) Postupné umisťování (heuristika)
// -------------------------------------------------------------

bool incremental_safe_placement(signed long *result, unsigned long *pocet_kroku) {
    /*
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

        if (success && is_valid(result))
            return true;
    }
    return false;
    */
    return false;
}

// -------------------------------------------------------------
// Pomocná funkce pro měření času
// -------------------------------------------------------------

void postup(bool (*func)(signed long*, unsigned long*), signed long *queens, const char *name) {
    unsigned long pocet_kroku = 0;
    printf("%s\nPočkejte prosím...", name);
    bool found = func(queens, &pocet_kroku);

    printf(" Hotovo, bylo potřeba vykonat %lu kroků.\n", pocet_kroku);
    if (found) {
        printf("Nalezeno řešení: ");
        if (hrana_sachovnice <= 15) print_solution(queens);
    } else {
        printf("Nebylo nalezeno řešení.\n");
    }
    printf("\n");
}

// -------------------------------------------------------------
// Hlavní funkce
// -------------------------------------------------------------

int main(int argc, char **argv) {

    if (argc > 2) {
        fprintf(stderr, "Použití: %s [<hrana_sachovnice>]", *argv);
        return 1;
    }

    if (argc < 2) hrana_sachovnice = VYCHOZI_HRANA_SACHOVNICE;
    else hrana_sachovnice = strtol(argv[1], NULL, 10);

    hrana_sachovnice_krat_3 = hrana_sachovnice * 3;

    printf("Porovnání 4 algoritmů pro problém 8 královen:\n\n");


    signed long *queens[POCET_ALGORITMU];
    for (int i=0;i < POCET_ALGORITMU;i++) queens[i] = malloc(hrana_sachovnice);

    postup(brute_force, queens[0], "1) Brute force");
    postup(dfs_backtrack, queens[1], "2) DFS/backtracking");
    postup(random_search, queens[2], "3) Náhodné hledání");
    postup(incremental_safe_placement, queens[3], "4) Postupné umisťování");

    for (int i=0;i < POCET_ALGORITMU;i++) free(queens[i]);

    return 0;
}
