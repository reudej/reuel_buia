#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <flint/fmpz.h>
#include <arf.h>
#include <signal.h>
//#include "bitptr.h"
//#include "bitptralloc.h"

//Platform-specific

#ifdef _WIN32
#define _WIN32_WINNT 0x0601 // Windows 7+
#include <windows.h>

#endif

#define VYCHOZI_HRANA_SACHOVNICE 1000000
#define POCET_ALGORITMU 5

signed long hrana_sachovnice;
signed long delka_diag_pole;
volatile sig_atomic_t ctrlc = 0;

fmpz_t celk_pocet_stavu;

// -------------------------------------------------------------
// Pomocné funkce
// -------------------------------------------------------------

void alokacni_chyba() {
    perror("Došlo k chybě při alokaci paměti.\n");
}

/*
static inline bool is_valid(const signed long *queens) {
    bool vrat = true;
    signed long n_m1 = hrana_sachovnice-1;

#define POCET_POLI_K_UVOLNENI 3
    bitptr_t diag1 = {0,0};
    bitptr_t diag2 = {0,0};
    bitptr_t sloupce_obsazeno = {0,0};
    bitptr_t* k_uvolneni[POCET_POLI_K_UVOLNENI] = {&diag1, &diag2, &sloupce_obsazeno};

    if (!bitptr_alloc(&diag1, delka_diag_pole)) goto alokacni_chyba_l;
    if (!bitptr_alloc(&diag2, delka_diag_pole)) goto alokacni_chyba_l;
    if (!bitptr_alloc(&sloupce_obsazeno, hrana_sachovnice)) goto alokacni_chyba_l;

    printf("0\n");
    bitptr_memclr(diag1, delka_diag_pole);
    printf("0\n");
    bitptr_memclr(diag2, delka_diag_pole);
    printf("0\n");
    bitptr_memclr(sloupce_obsazeno, hrana_sachovnice);
    printf("0\n");

    bitptr_t sloupce_obsazeno_ = sloupce_obsazeno;

    for (signed long row = 0; row < hrana_sachovnice && !ctrlc; row++,ppbitptr(&sloupce_obsazeno_)) {
        signed long col = queens[row];
        signed long diag1_i = row+col;
        signed long diag2_i = n_m1 + row - col;

        if (bitAt(sloupce_obsazeno_)) {vrat = false; break;}
        if (bitarr_get(diag1, diag1_i)) {vrat = false; break;}
        if (bitarr_get(diag2, diag2_i)) {vrat = false; break;}

        setBitAt(sloupce_obsazeno_, true);
        bitarr_set(diag1, diag1_i, true);
        bitarr_set(diag2, diag2_i, true);
    }

    end:
    for (int i=0;i < POCET_POLI_K_UVOLNENI;i++) { printf("%d\n", i); if (k_uvolneni[i]->byte) bitptr_free_(*(k_uvolneni[i])); else break; }
    if (ctrlc) return false;
    return vrat;

    alokacni_chyba_l:
    alokacni_chyba();
    vrat = false;
    goto end;
}
*/

void print_solution(const signed long *queens) {
    printf("[");
    for (signed long i = 0; i < hrana_sachovnice; i++) {
        printf("%ld", queens[i]);
        if (i < hrana_sachovnice - 1) printf(", ");
    }
    printf("]\n");
}

char* vetsina_alg_zjisti_celk_pocet_stavu() {
    fmpz_init(celk_pocet_stavu);
    fmpz_set_ui(celk_pocet_stavu, hrana_sachovnice);
    fmpz_pow_ui(celk_pocet_stavu, celk_pocet_stavu, hrana_sachovnice);
    return fmpz_get_str(NULL, 10, celk_pocet_stavu);
}


// -------------------------------------------------------------
// 4) Univerzální řešení
// -------------------------------------------------------------
signed char universal_solution(signed long *queens, unsigned long *pocet_prozk_stavu) {
    *pocet_prozk_stavu = 1; // žádné prozkoumávání, pouze výpočet

    if (hrana_sachovnice == 1) { queens[0] = 0; return 0; }
    if (hrana_sachovnice <= 3) return -1;

    signed long idx = 0;

    // Ahrens/Pauls method
    if (hrana_sachovnice % 6 == 2) {
        // Sequence: 1, 3, ..., N-1, 0, 2, ..., N-4, N-2, N-5
        for (long i = 1; i < hrana_sachovnice && !ctrlc; i += 2) queens[idx++] = i;          // odd rows first
        if (ctrlc) return -1;
        for (long i = 0; i < hrana_sachovnice - 4 && !ctrlc ; i += 2) queens[idx++] = i;      // even rows
        if (ctrlc) return -1;
        long last1 = hrana_sachovnice - 2;
        long last2 = hrana_sachovnice - 5;
        if (last1 >= 0) queens[idx++] = last1;
        if (last2 >= 0) queens[idx++] = last2;

    } else if (hrana_sachovnice % 6 == 3) {
        // Sequence: 2, 4, ..., N-1, 1, 3, ..., N-4, 0, N-2
        for (long i = 2; i < hrana_sachovnice && !ctrlc; i += 2) queens[idx++] = i;          // even (starting from 2)
        if (ctrlc) return -1;
        for (long i = 1; i < hrana_sachovnice - 1 && !ctrlc; i += 2) queens[idx++] = i;      // odd rows
        if (ctrlc) return -1;
        long last1 = 0;
        long last2 = hrana_sachovnice - 2;
        queens[idx++] = last1;
        queens[idx++] = last2;

    } else {
        // Simple case: odd rows then even rows
        printf("0\n");
        for (long i = 1; i < hrana_sachovnice && !ctrlc; i += 2) queens[idx++] = i;
        printf("0\n");
        if (ctrlc) return -1;
        for (long i = 0; i < hrana_sachovnice && !ctrlc; i += 2) queens[idx++] = i;
        printf("0\n");
        if (ctrlc) return -1;
    }
    return 0;
}

// -------------------------------------------------------------
// Zachycení signálů
// -------------------------------------------------------------


#ifdef _WIN32
BOOL WINAPI console_handler(DWORD signal) {
    switch (signal) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            ctrlc = 1;
            return TRUE; // signal zpracován
        default:
            return FALSE;
    }
}
#else
void handle_sigint(int sig) {
    ctrlc = 1;
}
#endif

// -------------------------------------------------------------
// Pomocná funkce pro vypsání informací o výsledku algoritmu
// -------------------------------------------------------------

void alg_info(signed char (*func)(signed long*, unsigned long*), signed long *queens, const char *name) {
    unsigned long pocet_prozk_stavu = 0;

    printf("%s\nPočkejte prosím... ", name);
    fflush(stdout);
    signed char found = func(queens, &pocet_prozk_stavu);

    arf_t podil;
    arf_init(podil);
    arf_set_ui(podil, pocet_prozk_stavu);
    arf_div_fmpz(podil, podil, celk_pocet_stavu, 16, ARF_RND_NEAR);
    arf_mul_ui(podil, podil, 100, 16, ARF_RND_NEAR);
    char *str_proc_podil = arf_get_str(podil, 2);

    if (found <= 0) printf(" Hotovo, prozkoumáno %lu stavů, což je %s%% z celkového počtu stavů.\n", pocet_prozk_stavu, str_proc_podil);

    arf_clear(podil);
    free(str_proc_podil);

    if (!found) {
        printf("Nalezeno řešení: ");
        print_solution(queens);
    } else if (found < 0) {
        printf("Nebylo nalezeno řešení.\n");
    } else if (found == 1) {
        alokacni_chyba();
    } else if (found == 2) {
        perror("Došlo k chybě při vytváření vlákna.\n");
    } else {
        fprintf(stderr, "Neznámá chyba: %d", found);
    }
    printf("\n");

    if (ctrlc) exit(3);
}

// -------------------------------------------------------------
// Nápověda
// -------------------------------------------------------------

int napoveda(char* arg0) {
    fprintf(stderr, "Použití: %s [<hrana_sachovnice>]", arg0);
    return 1;
}

// -------------------------------------------------------------
// Hlavní funkce
// -------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc > 2) return napoveda(*argv);

    if (argc < 2) hrana_sachovnice = VYCHOZI_HRANA_SACHOVNICE;
    else {
        char* ptr;
        hrana_sachovnice = strtol(argv[1], &ptr, 10);
        if (*ptr) return napoveda(*argv);
    }

#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#else
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);
#endif

    delka_diag_pole = hrana_sachovnice * 2 - 1;

    printf("Porovnání %d algoritmů pro problém %ld královen:\n\n", POCET_ALGORITMU, hrana_sachovnice);

    signed long *queens = malloc(hrana_sachovnice * sizeof(signed long));
    if (!queens) {
        alokacni_chyba();
        return 2;
    }

    char* vetsina_alg_celk_pocet_stavu = vetsina_alg_zjisti_celk_pocet_stavu();

    printf("Celkový počet stvavů: %s\n", vetsina_alg_celk_pocet_stavu);

    alg_info(universal_solution, queens, "5) Univerzální řešení");

    free(vetsina_alg_celk_pocet_stavu);
    fmpz_clear(celk_pocet_stavu);

    free(queens);

    if (ctrlc) return 3;
    return 0;
}
