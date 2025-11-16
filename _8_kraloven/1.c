#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "bitptr.h"
#include "bitptralloc.h"

#define VYCHOZI_HRANA_SACHOVNICE 8
#define POCET_ALGORITMU 4
#define RANDOM_SEARCH_KOLIKRAT 10 //Počet náhodných hledání pro zprůměrování počtu kroků

signed long hrana_sachovnice;

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
    return 0;
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
    return 0;
}

signed char brute_force(signed long *result, unsigned long *pocet_kroku) {
    for (int i = 0; i < hrana_sachovnice; i++) result[i] = i;

    do {
        if (is_valid(result)) return 0;
    } while (next_permutation(result));

    return false;
}

// -------------------------------------------------------------
// 2) DFS / backtracking
// -------------------------------------------------------------

signed char dfs_backtrack(signed long *queens, unsigned long *pocet_kroku) {
    signed long row = 0;
    signed long col = 0;
    while (1) {
        for (; col < hrana_sachovnice; col++) {
            *pocet_kroku += 3;
            bool safe = 0;
            for (signed long r = 0; r < row; r++) {
                signed long c = queens[r];
                if (c == col || labs(row - r) == labs(col - c)) {
                    safe = false;
                    break;
                }
            }
            if (safe) {
                queens[row++] = col;
                if (row == hrana_sachovnice) return 0;
                col = 0;
            }
        }
        if (--row < 0) return false;
        col = queens[row] + 1;
    }
}


signed char dfs_backtrack_optimized(signed long *queens, unsigned long *pocet_kroku) {
    signed long row = 0;
    signed long col = 0;
    bool vrat;
    while (1) {
        for (; col < hrana_sachovnice; col++) {
            *pocet_kroku += 3;
            bool safe = 0;
            for (signed long r = 0; r < row; r++) {
                signed long c = queens[r];
                if (c == col || labs(row - r) == labs(col - c)) {
                    safe = false;
                    break;
                }
            }
            if (safe) {
                queens[row++] = col;
                if (row == hrana_sachovnice) return 0;
                col = 0;
            }
        }
        if (--row < 0) return false;
        col = queens[row] + 1;
    }
    end_t:
    return 0;
    end_f:
    return false;
}

// -------------------------------------------------------------
// 3) Náhodné hledání
// -------------------------------------------------------------

signed char random_search(signed long *result, unsigned long *pocet_kroku) {
    while (1) {
        for (int i = 0; i < hrana_sachovnice;i++) result[i] = rand() % hrana_sachovnice;
        if (is_valid(result))
            return 0;
    }
    return false;
}

// -------------------------------------------------------------
// 4) Univerzální řešení
// -------------------------------------------------------------

signed char universal_solution(signed long *queens, unsigned long *pocet_kroku) {
    if (hrana_sachovnice == 1) { queens[0] = 0; return 0; }
    if (hrana_sachovnice == 2 || hrana_sachovnice == 3) return false;

    unsigned long idx = 0;
    if (hrana_sachovnice % 6 == 2) {
        int seq1[] = {2,0,3,1};
        int len1 = 4;
        for (int i = 0; i < len1 && seq1[i] <= hrana_sachovnice; i++) queens[idx++] = seq1[i];
        for (int i = 5; i <= hrana_sachovnice; i += 2) queens[idx++] = i;
        for (int i = 4; i <= hrana_sachovnice-1; i += 2) queens[idx++] = i;
    } else if (hrana_sachovnice % 6 == 3) {
        for (int i = 1; i <= hrana_sachovnice-1; i += 2) queens[idx++] = i;
        for (int i = 0; i <= hrana_sachovnice-2; i += 2) queens[idx++] = i;
        queens[idx++] = hrana_sachovnice - 1;
    } else {
        for (int i = 1; i <= hrana_sachovnice; i += 2) queens[idx++] = i;
        for (int i = 0; i <= hrana_sachovnice; i += 2) queens[idx++] = i;
    }
    return 0;
}

// -------------------------------------------------------------
// Pomocná funkce pro vypsání informací o výsledku algoritmu
// -------------------------------------------------------------

void alg_info(signed char (*func)(signed long*, unsigned long*), signed long *queens, const char *name) {
    unsigned long pocet_kroku = 0;
    printf("%s\nPočkejte prosím...", name);
    signed char found = func(queens, &pocet_kroku);

    printf(" Hotovo, bylo potřeba vykonat %lu kroků.\n", pocet_kroku);
    if (found) {
        printf("Nalezeno řešení: ");
        if (hrana_sachovnice <= 15) print_solution(queens);
    } else if (found == -1) {
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

    //hrana_sachovnice_krat_3 = hrana_sachovnice * 3;

    printf("Porovnání 4 algoritmů pro problém 8 královen:\n\n");


    signed long *queens[POCET_ALGORITMU];
    for (int i=0;i < POCET_ALGORITMU;i++) queens[i] = malloc(hrana_sachovnice);

    alg_info(brute_force, queens[0], "1) Brute force");
    alg_info(dfs_backtrack, queens[1], "2) DFS/backtracking");
    alg_info(random_search, queens[2], "3) Náhodné hledání");
    alg_info(universal_solution, queens[3], "4) Postupné umisťování");

    for (int i=0;i < POCET_ALGORITMU;i++) free(queens[i]);

    return 0;
}
