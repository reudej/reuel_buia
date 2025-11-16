#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <flint/fmpz.h>
#include <arf.h>
#include "bitptr.h"
#include "bitptralloc.h"
#include "portable_atomic.h"

//Platform-specific

#ifdef _WIN32
#define _WIN32_WINNT 0x0601 // Windows 7+
#include <windows.h>
typedef HANDLE vlakno_t;
typedef DWORD WINAPI vlakno_vrat_t;
typedef DWORD WINAPI¨ vytvor_vlakno_chyba_t;

#else
#include <hwloc.h>
#include <signal.h>
#include <pthread.h>
typedef pthread_t vlakno_t;
typedef void* vlakno_vrat_t;
typedef int vytvor_vlakno_chyba_t;

#endif

#define VYCHOZI_HRANA_SACHOVNICE 8
#define POCET_ALGORITMU 5
#define RANDOM_SEARCH_KOLIKRAT 10000 //Počet náhodných hledání pro zprůměrování počtu kroků

signed long hrana_sachovnice;
signed long delka_diag_pole;
unsigned int pocet_fyzickych_jader=0;
unsigned long pocet_sloupcu_na_vlakno;
unsigned long pocet_sloupcu_na_posl_vlakno;
portable_atomic_int stop;

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
    for (signed long i = 0; i < hrana_sachovnice; i++) {
        printf("%ld", queens[i]);
        if (i < hrana_sachovnice - 1) printf(", ");
    }
    printf("]\n");
}

void alokacni_chyba() {
    perror("Došlo k chybě při alokaci paměti.\n");
}

// -------------------------------------------------------------
// Rozdělení práce mezi vlákna
// -------------------------------------------------------------

//Zjištění počtu fyzických jader
static inline void zjisti_pocet_fyzickych_jader() {
#ifdef _WIN32
    DWORD len = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buf = NULL;

    // Zjistit velikost bufferu
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);

    buf = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(len);
    if (!buf) {
        perror(" malloc failed\n");
        return;
    }

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
        perror(" GetLogicalProcessorInformationEx failed\n");
        free(buf);
        return;
    }

    int physicalCores = 0;

    BYTE *ptr = (BYTE *)buf;
    BYTE *end = ptr + len;

    while (ptr < end) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;

        if (info->Relationship == RelationProcessorCore)
            physicalCores++;

        ptr += info->Size;
    }

    pocet_fyzickych_jader = physicalCores;

    free(buf);
#else
    hwloc_topology_t topology;
    int rc;

    rc = hwloc_topology_init(&topology);
    if (rc < 0) {
        perror(" hwloc_topology_init failed\n");
        return;
    }

    rc = hwloc_topology_load(topology);
    if (rc < 0) {
        perror(" hwloc_topology_load failed\n");
        hwloc_topology_destroy(topology);
        return;
    }

    int count = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);

    // Fallback pokud HWLOC_OBJ_CORE není supportované
    if (count <= 0)
        count = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);

    hwloc_topology_destroy(topology);
    pocet_fyzickych_jader = count;
#endif
}

typedef struct {
    char (*callback)(signed long*, unsigned long*, signed long, signed long);
    signed long *queens;
    unsigned long *pocet_prozk_stavu;
    signed long start;
    signed long stop;
    char *chyba;
} thread_arg_t;

vlakno_vrat_t callback_wrapper(void *arg_void) {
    thread_arg_t *arg = (thread_arg_t*)arg_void;
    *arg->chyba = arg->callback(arg->queens, arg->pocet_prozk_stavu, arg->start, arg->stop);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}
static inline bool vytvor_vlakno(char (*callback)(signed long*, unsigned long*, signed long, signed long), signed long *queens, unsigned long *pocet_prozk_stavu, char *chyba, signed long start, signed long stop, vlakno_t *vlakno) {
    thread_arg_t arg;
    arg.callback = callback;
    arg.queens = queens;
    arg.pocet_prozk_stavu = pocet_prozk_stavu;
    arg.start = start;
    arg.stop = stop;
    arg.chyba = chyba;
#ifdef _WIN32
    vlakno = CreateThread(NULL, 0, callback_wrapper, &arg, 0, NULL);
    vytvor_vlakno_chyba_t chyba_vytvareni = vlakno;
    if (chyba_vytvareni) {
#else
    vytvor_vlakno_chyba_t chyba_vytvareni = pthread_create(vlakno, NULL, callback_wrapper, &arg);
    if (!chyba_vytvareni) {
#endif
        printf("Chyba při vytváření vlákna\n");
        return false;
    }
    return true;
}
static inline void zastav_vlakna() {
    portable_atomic_store(&stop, 1);
}
static inline char rozdel_praci(char (*callback)(signed long*, unsigned long*, signed long, signed long), signed long *queens, unsigned long *pocet_prozk_stavu) {
    signed long start = 0;
    signed long stop = pocet_sloupcu_na_vlakno;

    vlakno_t vlakna[pocet_fyzickych_jader];
    unsigned long pocet_prozk_stavu_vlaknem[pocet_fyzickych_jader];
    char chyba_vlakna[pocet_fyzickych_jader];
    signed long *queens_arr = malloc(pocet_fyzickych_jader * hrana_sachovnice);
    vlakno_t *vlakno = vlakna;
    unsigned long *pocet_prozk_stavu_vlaknem_ = pocet_prozk_stavu_vlaknem;
    char *chyba_vlakna_ = chyba_vlakna;
    signed long *queens_ = queens_arr;
    int index=0;
    bool spusteno_bez_chyby = true;
    for (; index < pocet_fyzickych_jader-1 ; index++,vlakno++,pocet_prozk_stavu_vlaknem_++,chyba_vlakna_++,queens_+=hrana_sachovnice,start=stop,stop+=pocet_sloupcu_na_vlakno) {
        if (!vytvor_vlakno(callback, queens_, pocet_prozk_stavu_vlaknem_, chyba_vlakna_, start, stop, vlakno)) {
            zastav_vlakna();
            spusteno_bez_chyby = false;
            break;
        }
    }
    if (spusteno_bez_chyby) {
        stop += pocet_sloupcu_na_posl_vlakno;
        if (!vytvor_vlakno(callback, queens_, pocet_prozk_stavu_vlaknem_, chyba_vlakna_, start, stop, vlakno)) zastav_vlakna();
    }

    vlakno = vlakna;
    pocet_prozk_stavu_vlaknem_ = pocet_prozk_stavu_vlaknem;
    chyba_vlakna_ = chyba_vlakna;
    queens_ = queens_arr;

    signed int found = -1;
    for (int i=0;i < index;i++,vlakno++,pocet_prozk_stavu_vlaknem_++,chyba_vlakna_++,queens_+=hrana_sachovnice) {
#ifdef _WIN32
        WaitForSingleObject(vlakno, INFINITE);
#else
        pthread_join(*vlakno, NULL);
#endif

        if (spusteno_bez_chyby) {
            *pocet_prozk_stavu += *pocet_prozk_stavu_vlaknem_;
            if (!*chyba_vlakna_ && found) {
                found = 0;
                memcpy(queens, queens_, hrana_sachovnice * sizeof(long));
            } else if (*chyba_vlakna_ && found < 0) {
                found = *chyba_vlakna_;
            }
        }
    }

    free(queens_arr);

    if (!spusteno_bez_chyby) return 2;
    return found;
}

// -------------------------------------------------------------
// 1) Brute force – zkouší všechny permutace
// -------------------------------------------------------------

static inline bool next_permutation(signed long *arr) {
    signed long i = hrana_sachovnice - 2;
    while (i >= 0 && arr[i] > arr[i + 1]) i--;
    if (i < 0) return false;
    signed long j = hrana_sachovnice - 1;
    while (arr[j] < arr[i]) j--;
    signed long tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    for (signed long a = i + 1, b = hrana_sachovnice - 1; a < b; a++, b--) {
        tmp = arr[a]; arr[a] = arr[b]; arr[b] = tmp;
    }
    return 0;
}

signed char brute_force(signed long *result, unsigned long *pocet_kroku) {
    for (signed long i = 0; i < hrana_sachovnice; i++) result[i] = i;

    do {
        if (is_valid(result)) return 0;
    } while (next_permutation(result));

    return -1;
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
                if (row == hrana_sachovnice) return 0;
                col = 0;
            }
        }
        if (--row < 0) return -1;
        col = queens[row] + 1;
    }
}

signed char dfs_backtrack_optimized(signed long *queens, unsigned long *pocet_kroku) {
    signed long row = 0;
    signed long col = 0;

    //Pole pro uložení obsazenosti diagonál
#define POCET_b_POLI_K_UVOLNENI 3
    bitptr_t diag1;
    bitptr_t diag2;
    bitptr_t sloupce_obsazeno;

    //Inicializace pole polí, které bude potřeba uvolnit.
    bitptr_t *k_uvolneni[POCET_b_POLI_K_UVOLNENI] = {&diag1, &diag2, &sloupce_obsazeno};
    for (int i=0;i < POCET_b_POLI_K_UVOLNENI;i++) {
        k_uvolneni[i]->byte = 0;
    }

    if (!bitptr_alloc(&diag1, delka_diag_pole)) goto alokacni_chyba_l;
    if (!bitptr_alloc(&diag2, delka_diag_pole)) goto alokacni_chyba_l;
    if (!bitptr_alloc(&sloupce_obsazeno, hrana_sachovnice)) goto alokacni_chyba_l;
    bitptr_memclr(diag1, delka_diag_pole);
    bitptr_memclr(diag2, delka_diag_pole);
    bitptr_memclr(sloupce_obsazeno, hrana_sachovnice);

    //Posuvné ukazatele
    bitptr_t sloupce_obsazeno_;
    bitptr_t diag1_;
    bitptr_t diag2_;

    char vrat = 0;
    unsigned long n_div_2_ceil = (hrana_sachovnice>>2) + (hrana_sachovnice&1);
    unsigned long n_min_1 = hrana_sachovnice -1;

    while (1) {
        zacatek_while:
        sloupce_obsazeno_ = sloupce_obsazeno;
        diag1_ = bitptrAdd_return(diag1, row+col);
        diag2_ = bitptrAdd_return(diag2, row-col + hrana_sachovnice-1);
        for (;  col < hrana_sachovnice && (row < n_min_1 || n_div_2_ceil)  ;  col++, ppbitptr(&diag1_), mmbitptr(&diag2_)) {
            //Kontrola obsazenosti
            if (bitAt(sloupce_obsazeno_)) continue;
            if (bitAt(diag1_)) continue;
            if (bitAt(diag2_)) continue;

            //Zápis obsazenosti
            setBitAt(sloupce_obsazeno_, true);
            setBitAt(diag1_, true);
            setBitAt(diag2_, true);

            queens[row++] = col;
            if (row == hrana_sachovnice) goto end;
            ppbitptr(&sloupce_obsazeno_);
            col = 0;
            goto zacatek_while;
        }
        if (--row < 0) {vrat = -1; goto end;}
        col = ++queens[row];
    }

    alokacni_chyba_l:
    alokacni_chyba();
    vrat = 2;

    end:
    for (int i=0;i < POCET_b_POLI_K_UVOLNENI;i++) {
        if (k_uvolneni[i]->byte) bitptr_free_(*k_uvolneni[i]);
        else break;
    }
    return vrat;
}

// -------------------------------------------------------------
// 3) Náhodné hledání
// -------------------------------------------------------------

signed char random_search(signed long *result, unsigned long *pocet_kroku) {
    while (1) {
        for (signed long i = 0; i < hrana_sachovnice;i++) result[i] = rand() % hrana_sachovnice;
        if (is_valid(result))
            return 0;
    }
    return -1;
}

// -------------------------------------------------------------
// 4) Univerzální řešení
// -------------------------------------------------------------

signed char universal_solution(signed long *queens, unsigned long *pocet_kroku) {
    if (hrana_sachovnice == 1) { queens[0] = 0; return 0; }
    if (hrana_sachovnice == 2 || hrana_sachovnice == 3) return -1;

    unsigned long idx = 0;
    if (hrana_sachovnice % 6 == 2) {
        int seq1[] = {2,0,3,1};
        int len1 = 4;
        for (int i = 0; i < len1 && seq1[i] <= hrana_sachovnice; i++) queens[idx++] = seq1[i];
        for (signed long i = 5; i <= hrana_sachovnice; i += 2) queens[idx++] = i;
        for (signed long i = 4; i <= hrana_sachovnice-1; i += 2) queens[idx++] = i;
    } else if (hrana_sachovnice % 6 == 3) {
        for (signed long i = 1; i <= hrana_sachovnice-1; i += 2) queens[idx++] = i;
        for (signed long i = 0; i <= hrana_sachovnice-2; i += 2) queens[idx++] = i;
        queens[idx++] = hrana_sachovnice - 1;
    } else {
        for (signed long i = 1; i <= hrana_sachovnice; i += 2) queens[idx++] = i;
        for (signed long i = 0; i <= hrana_sachovnice; i += 2) queens[idx++] = i;
    }
    return 0;
}

// -------------------------------------------------------------
// Pomocná funkce pro vypsání informací o výsledku algoritmu
// -------------------------------------------------------------

void alg_info(signed char (*func)(signed long*, unsigned long*), signed long *queens, const char *name) {
    unsigned long pocet_prozk_stavu = 0;

    printf("%s\nPočkejte prosím...", name);
    signed char found = func(queens, &pocet_prozk_stavu);

    fmpz_t celk_pocet_stavu;
    fmpz_init((celk_pocet_stavu));
    fmpz_set_ui(celk_pocet_stavu, hrana_sachovnice);
    fmpz_pow_ui(celk_pocet_stavu, celk_pocet_stavu, hrana_sachovnice);
    char *str_celk_pocet = fmpz_get_str(NULL, 10, celk_pocet_stavu);

    arf_t podil;
    arf_init(podil);
    arf_set_ui(podil, pocet_prozk_stavu);
    arf_div_fmpz(podil, podil, celk_pocet_stavu, 16, ARF_RND_NEAR);
    arf_mul_ui(podil, podil, 10000, 16, ARF_RND_NEAR);
    char *str_proc_podil = arf_get_str(podil, 2);

    printf(" Hotovo, bylo potřeba prozkoumat %lu stavů, což je %s%%.\n z celkových %s stavů", pocet_prozk_stavu, str_proc_podil, str_celk_pocet);

    fmpz_clear(celk_pocet_stavu);
    arf_clear(podil);
    free(str_celk_pocet);
    free(str_proc_podil);

    if (!found) {
        printf("Nalezeno řešení: ");
        if (hrana_sachovnice <= 15) print_solution(queens);
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
}

// -------------------------------------------------------------
// Nápověda
// -------------------------------------------------------------

int napoveda(char* arg0) {
    fprintf(stderr, "Použití: %s [<hrana_sachovnice>]", arg0);
    return 1;
}

// -------------------------------------------------------------
// Zachycení Ctrl+C
// -------------------------------------------------------------

volatile sig_atomic_t ctrlc = 0;

#ifdef _WIN32
BOOL WINAPI console_handler(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        ctrlc = 1;
        return TRUE; // signal zpracován
    }
    return FALSE;
}
#else
void handle_sigint(int sig) {
    ctrlc = 1;
}
#endif

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
#endif

    portable_atomic_init(&stop, 0); //Inicializujeme proměnnou pro zastevení všech vláken kromě hlavního.

    delka_diag_pole = hrana_sachovnice * 2 - 1;

    printf("Porovnání %d algoritmů pro problém 8 královen:\n\n", POCET_ALGORITMU);

    printf("Probíhá automatické zjišťování počtu fyzických jader...");
    zjisti_pocet_fyzickych_jader();
    if (pocet_fyzickych_jader == 0) {
        printf("Automatické zjištění selhalo.\n");
        do printf("Zadejte prosím počet fyzických jader ručně: "); while (scanf("%d", &pocet_fyzickych_jader) <= 0 && !ctrlc);
    }

    pocet_sloupcu_na_vlakno = hrana_sachovnice/pocet_fyzickych_jader;
    pocet_sloupcu_na_posl_vlakno = hrana_sachovnice%pocet_fyzickych_jader;

    signed long *queens[POCET_ALGORITMU];
    for (int i=0;i < POCET_ALGORITMU;i++) {
        queens[i] = malloc(hrana_sachovnice);
        if (!queens[i]) {
            alokacni_chyba();
            return 2;
        }
    }

    alg_info(brute_force, queens[0], "1) Brute force");
    alg_info(dfs_backtrack, queens[1], "2) DFS s backtrackingem");
    alg_info(dfs_backtrack_optimized, queens[2], "3) DFS s backtrackingem - optimalizováno");
    alg_info(random_search, queens[3], "4) Náhodné hledání");
    alg_info(universal_solution, queens[4], "5) Univerzální řešení");

    for (int i=0;i < POCET_ALGORITMU;i++) free(queens[i]);

    return 0;
}
