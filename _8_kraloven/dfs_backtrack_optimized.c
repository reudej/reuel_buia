#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <flint/fmpz.h>
#include <arf.h>
#include <signal.h>
#include "bitptr.h"
#include "bitptralloc.h"
#include "portable_atomic.h"

//Platform-specific

#ifdef _WIN32
#define _WIN32_WINNT 0x0601 // Windows 7+
#include <windows.h>
typedef HANDLE vlakno_t;
typedef DWORD WINAPI vlakno_vrat_t;
typedef DWORD WINAPI vytvor_vlakno_chyba_t;

#else
#include <hwloc.h>
#include <pthread.h>
typedef pthread_t vlakno_t;
typedef void* vlakno_vrat_t;
typedef int vytvor_vlakno_chyba_t;

#endif

#define VYCHOZI_HRANA_SACHOVNICE 28
#define POCET_ALGORITMU 5
#define RANDOM_SEARCH_KOLIKRAT 10000 //Počet náhodných hledání pro zprůměrování počtu kroků

signed long hrana_sachovnice;
signed long delka_diag_pole;
unsigned long n_div_2_ceil;
unsigned long n_min_1;
unsigned long n_min_2;
unsigned int pocet_fyzickych_jader=0;
unsigned long pocet_sloupcu_na_vlakno;
unsigned long pocet_sloupcu_na_posl_vlakno;
portable_atomic_int stop_vlaken;

fmpz_t celk_pocet_stavu;

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
    for (signed long i = 0; i < hrana_sachovnice; i++) {
        printf("%ld", queens[i]);
        if (i < hrana_sachovnice - 1) printf(", ");
    }
    printf("]\n");
}

void alokacni_chyba() {
    perror("Došlo k chybě při alokaci paměti.\n");
}

char* vetsina_alg_zjisti_celk_pocet_stavu() {
    fmpz_init(celk_pocet_stavu);
    fmpz_set_ui(celk_pocet_stavu, hrana_sachovnice);
    fmpz_pow_ui(celk_pocet_stavu, celk_pocet_stavu, hrana_sachovnice);
    return fmpz_get_str(NULL, 10, celk_pocet_stavu);
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
        perror(" malloc failed.\n");
        return;
    }

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
        perror(" GetLogicalProcessorInformationEx failed.\n");
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
        perror(" hwloc_topology_init failed.\n");
        return;
    }

    rc = hwloc_topology_load(topology);
    if (rc < 0) {
        perror(" hwloc_topology_load failed.\n");
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
    signed char (*callback)(signed long*, unsigned long*, signed long, signed long);
    signed long *queens;
    unsigned long *pocet_prozk_stavu;
    signed long start;
    signed long stop;
    signed char *chyba;
} thread_arg_t;

vlakno_vrat_t callback_wrapper(void *arg_void) {
    thread_arg_t *arg = arg_void;
    *arg->chyba = arg->callback(arg->queens, arg->pocet_prozk_stavu, arg->start, arg->stop);
    free(arg);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}
static inline unsigned short vytvor_vlakno(signed char (*callback)(signed long*, unsigned long*, signed long, signed long), signed long *queens, unsigned long *pocet_prozk_stavu, signed char *chyba, signed long start, signed long stop, vlakno_t *vlakno) {
    thread_arg_t *arg = malloc(sizeof(thread_arg_t));
    if (!arg) return 1;
    arg->callback = callback;
    arg->queens = queens;
    arg->pocet_prozk_stavu = pocet_prozk_stavu;
    arg->start = start;
    arg->stop = stop;
    arg->chyba = chyba;
#ifdef _WIN32
    *vlakno = CreateThread(NULL, 0, callback_wrapper, arg, 0, NULL);
    vytvor_vlakno_chyba_t chyba_vytvareni = vlakno;
    if (!chyba_vytvareni) {
#else
    vytvor_vlakno_chyba_t chyba_vytvareni = pthread_create(vlakno, NULL, callback_wrapper, arg);
    if (chyba_vytvareni) {
#endif
        free(arg);
        return 2;
    }
    return 0;
}

static inline void zastav_vlakna() {
    portable_atomic_store(&stop_vlaken, 1);
}
static inline bool mam_skoncit() {
    return (bool)portable_atomic_load(&stop_vlaken);
}

static inline signed char rozdel_praci(signed char (*callback)(signed long*, unsigned long*, signed long, signed long), signed long *queens, unsigned long *pocet_prozk_stavu) {
    signed long start = 0;
    signed long stop = pocet_sloupcu_na_vlakno;

    vlakno_t vlakna[pocet_fyzickych_jader];
    unsigned long pocet_prozk_stavu_vlaknem[pocet_fyzickych_jader];
    signed char chyba_vlakna[pocet_fyzickych_jader];
    signed long *queens_arr = malloc(pocet_fyzickych_jader * hrana_sachovnice * sizeof(signed long));
    vlakno_t *vlakno = vlakna;
    unsigned long *pocet_prozk_stavu_vlaknem_ = pocet_prozk_stavu_vlaknem;
    signed char *chyba_vlakna_ = chyba_vlakna;
    signed long *queens_ = queens_arr;

    //Inicializujeme pole počtu prozkoumaých stavů pro každé vlákno.
    for (long i=0;i < pocet_fyzickych_jader;i++) pocet_prozk_stavu_vlaknem[i] = 0;

    int index=0;
    bool spusteno_bez_chyby = true;
    for (; index < pocet_fyzickych_jader-1 ; index++,vlakno++,pocet_prozk_stavu_vlaknem_++,chyba_vlakna_++,queens_+=hrana_sachovnice,start=stop,stop+=pocet_sloupcu_na_vlakno) {
        if (vytvor_vlakno(callback, queens_, pocet_prozk_stavu_vlaknem_, chyba_vlakna_, start, stop, vlakno)) {
            zastav_vlakna();
            spusteno_bez_chyby = false;
            break;
        }
    }
    if (spusteno_bez_chyby) {
        stop += pocet_sloupcu_na_posl_vlakno;
        if (vytvor_vlakno(callback, queens_, pocet_prozk_stavu_vlaknem_, chyba_vlakna_, start, stop, vlakno)) zastav_vlakna();
        else index++;
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
                memcpy(queens, queens_, hrana_sachovnice * sizeof(signed long));
            } else if (*chyba_vlakna_ && found) {
                found = *chyba_vlakna_;
            }
        }
    }

    free(queens_arr);

    portable_atomic_store(&stop_vlaken, 0);

    if (!spusteno_bez_chyby) return 2;
    return found;
}

// -------------------------------------------------------------
// 2) DFS / backtracking
// -------------------------------------------------------------

char* dfs_backtrack_zjisti_celk_pocet_stavu() {
    fmpz_t celk_pocet_stavu;
    fmpz_init(celk_pocet_stavu);
    fmpz_ui_pow_ui(celk_pocet_stavu, hrana_sachovnice, hrana_sachovnice);
    fmpz_sub_si(celk_pocet_stavu, celk_pocet_stavu, 1);
    fmpz_mul_si(celk_pocet_stavu, celk_pocet_stavu, hrana_sachovnice);

    fmpz_t jmenovatel;
    fmpz_init(jmenovatel);
    fmpz_set_si(jmenovatel, hrana_sachovnice);
    fmpz_sub_si(jmenovatel, jmenovatel, 1);

    fmpz_divexact(celk_pocet_stavu, celk_pocet_stavu, jmenovatel);
    fmpz_clear(jmenovatel);
    char* str = fmpz_get_str(NULL, 10, celk_pocet_stavu);
    fmpz_clear(celk_pocet_stavu);
    return str;
}

static inline void nastav_posuvne_ptr(bitptr_t *sloupce_obsazeno_, bitptr_t sloupce_obsazeno, bitptr_t *diag1_, bitptr_t diag1, bitptr_t *diag2_, bitptr_t diag2, signed long row, signed long col) {
    *sloupce_obsazeno_ = sloupce_obsazeno;
    *diag1_ = bitptrAdd_return(diag1, row+col);
    *diag2_ = bitptrAdd_return(diag2, row-col + hrana_sachovnice-1);
}

static inline signed char dfs_backtrack_optimized_core_row_2plus(signed long *queens, unsigned long *pocet_prozk_stavu, bitptr_t sloupce_obsazeno_, bitptr_t sloupce_obsazeno, bitptr_t diag1_, bitptr_t diag1, bitptr_t diag2_, bitptr_t diag2, bitptr_t *sloupce_obsazeno_bitptr_v, bitptr_t *diag1_bitptr_v, bitptr_t *diag2_bitptr_v) {
    signed long row = 2;
    signed long col = 0;
    while (!mam_skoncit()) {
        nastav_posuvne_ptr(&sloupce_obsazeno_, sloupce_obsazeno, &diag1_, diag1, &diag2_, diag2, row, col);
        for (;  col < hrana_sachovnice && !mam_skoncit() ;  col++, ppbitptr(&sloupce_obsazeno_), ppbitptr(&diag1_), mmbitptr(&diag2_), (*pocet_prozk_stavu)++) {
            //Kontrola obsazenosti
            if (bitAt(sloupce_obsazeno_)) continue;
            if (bitAt(diag1_)) continue;
            if (bitAt(diag2_)) continue;

            //Zápis obsazenosti
            setBitAt(sloupce_obsazeno_, true);
            setBitAt(diag1_, true);
            setBitAt(diag2_, true);

            //Zapíšeme si, kam jsme obsazenost zapsali, abychom zápis mohli jednoduše zrušit.
            sloupce_obsazeno_bitptr_v[row] = sloupce_obsazeno_;
            diag1_bitptr_v[row] = diag1_;
            diag2_bitptr_v[row] = diag2_;

            queens[row++] = col;
            if (row == hrana_sachovnice) return 0;
            col = 0;
            nastav_posuvne_ptr(&sloupce_obsazeno_, sloupce_obsazeno, &diag1_, diag1, &diag2_, diag2, row, col);
        }
        if (--row < 2) return -1;
        col = ++queens[row];

        //Zrušíme zápis obsazenosti.
        setBitAt(sloupce_obsazeno_bitptr_v[row], false);
        setBitAt(diag1_bitptr_v[row], false);
        setBitAt(diag2_bitptr_v[row], false);
    }
    return -1;
}
static inline signed char dfs_backtrack_optimized_core_row_1(signed long *queens, unsigned long *pocet_prozk_stavu, bitptr_t sloupce_obsazeno_, bitptr_t sloupce_obsazeno, bitptr_t diag1_, bitptr_t diag1, bitptr_t diag2_, bitptr_t diag2, bitptr_t *sloupce_obsazeno_bitptr_v, bitptr_t *diag1_bitptr_v, bitptr_t *diag2_bitptr_v) {
    const signed long row = 1;
    signed long col = 0;
    nastav_posuvne_ptr(&sloupce_obsazeno_, sloupce_obsazeno, &diag1_, diag1, &diag2_, diag2, row, col);
    for (;  col < n_div_2_ceil && !mam_skoncit()  ;  col++, ppbitptr(&sloupce_obsazeno_), ppbitptr(&diag1_), mmbitptr(&diag2_), (*pocet_prozk_stavu)++) {
        //Kontrola obsazenosti
        if (bitAt(sloupce_obsazeno_)) continue;
        if (bitAt(diag1_)) continue;
        if (bitAt(diag2_)) continue;

        //Zápis obsazenosti
        setBitAt(sloupce_obsazeno_, true);
        setBitAt(diag1_, true);
        setBitAt(diag2_, true);

        //Zapíšeme si, kam jsme obsazenost zapsali, abychom zápis mohli jednoduše zrušit.
        sloupce_obsazeno_bitptr_v[row] = sloupce_obsazeno_;
        diag1_bitptr_v[row] = diag1_;
        diag2_bitptr_v[row] = diag2_;

        queens[row] = col;
        if (dfs_backtrack_optimized_core_row_2plus(queens, pocet_prozk_stavu, sloupce_obsazeno_, sloupce_obsazeno, diag1_, diag1, diag2_, diag2, sloupce_obsazeno_bitptr_v, diag1_bitptr_v, diag2_bitptr_v)) {
            //Zrušíme zápis obsazenosti.
            setBitAt(sloupce_obsazeno_bitptr_v[row], false);
            setBitAt(diag1_bitptr_v[row], false);
            setBitAt(diag2_bitptr_v[row], false);
        }

        else return 0;
    }
    return -1;
}
static inline signed char dfs_backtrack_optimized_core_row_0(signed long *queens, unsigned long *pocet_prozk_stavu, signed long start, signed long stop,  bitptr_t sloupce_obsazeno_, bitptr_t sloupce_obsazeno, bitptr_t diag1_, bitptr_t diag1, bitptr_t diag2_, bitptr_t diag2, bitptr_t *sloupce_obsazeno_bitptr_v, bitptr_t *diag1_bitptr_v, bitptr_t *diag2_bitptr_v) {
    const signed long row = 0;
    signed long col = start;
    nastav_posuvne_ptr(&sloupce_obsazeno_, sloupce_obsazeno, &diag1_, diag1, &diag2_, diag2, row, col);
    for (;  col < stop && !mam_skoncit()  ;  col++, ppbitptr(&sloupce_obsazeno_), ppbitptr(&diag1_), mmbitptr(&diag2_), (*pocet_prozk_stavu)++) {
        //Kontrola obsazenosti
        if (bitAt(sloupce_obsazeno_)) continue;
        if (bitAt(diag1_)) continue;
        if (bitAt(diag2_)) continue;

        //Zápis obsazenosti
        setBitAt(sloupce_obsazeno_, true);
        setBitAt(diag1_, true);
        setBitAt(diag2_, true);

        //Zapíšeme si, kam jsme obsazenost zapsali, abychom zápis mohli jednoduše zrušit.
        sloupce_obsazeno_bitptr_v[row] = sloupce_obsazeno_;
        diag1_bitptr_v[row] = diag1_;
        diag2_bitptr_v[row] = diag2_;

        queens[row] = col;
        if (dfs_backtrack_optimized_core_row_1(queens, pocet_prozk_stavu, sloupce_obsazeno_, sloupce_obsazeno, diag1_, diag1, diag2_, diag2, sloupce_obsazeno_bitptr_v, diag1_bitptr_v, diag2_bitptr_v)) {
            //Zrušíme zápis obsazenosti.
            setBitAt(sloupce_obsazeno_bitptr_v[row], false);
            setBitAt(diag1_bitptr_v[row], false);
            setBitAt(diag2_bitptr_v[row], false);


        } else {
            zastav_vlakna();
            return 0;
        }
    }
    return -1;
}
static inline signed char dfs_backtrack_optimized_thread(signed long *queens, unsigned long *pocet_prozk_stavu, signed long start, signed long stop) {
    signed long row = 0;
    signed long col = 0;

    //Pole pro uložení obsazenosti diagonál
#define POCET_b_POLI_K_UVOLNENI 3
#define POCET_c_POLI_K_UVOLNENI 3
    bitptr_t diag1;
    bitptr_t diag2;
    bitptr_t sloupce_obsazeno;

    bitptr_t *diag1_bitptr_v = 0;
    bitptr_t *diag2_bitptr_v = 0;
    bitptr_t *sloupce_obsazeno_bitptr_v = 0;

    //Inicializace polí polí, které bude potřeba uvolnit.
    bitptr_t** k_uvolneni_c[POCET_c_POLI_K_UVOLNENI] = {&diag1_bitptr_v, &diag2_bitptr_v, &sloupce_obsazeno_bitptr_v};
    bitptr_t* k_uvolneni[POCET_b_POLI_K_UVOLNENI] = {&diag1, &diag2, &sloupce_obsazeno};
    for (int i=0;i < POCET_b_POLI_K_UVOLNENI;i++) {
        k_uvolneni[i]->byte = 0;
    }

    if (!bitptr_alloc(&diag1, delka_diag_pole)) goto alokacni_chyba_l;
    if (!bitptr_alloc(&diag2, delka_diag_pole)) goto alokacni_chyba_l;
    if (!bitptr_alloc(&sloupce_obsazeno, hrana_sachovnice)) goto alokacni_chyba_l;

    diag1_bitptr_v = malloc(hrana_sachovnice * sizeof(bitptr_t));
    if (!diag1_bitptr_v) goto alokacni_chyba_l;
    diag2_bitptr_v = malloc(hrana_sachovnice * sizeof(bitptr_t));
    if (!diag2_bitptr_v) goto alokacni_chyba_l;
    sloupce_obsazeno_bitptr_v = malloc(hrana_sachovnice * sizeof(bitptr_t));
    if (!sloupce_obsazeno_bitptr_v) goto alokacni_chyba_l;

    bitptr_memclr(diag1, delka_diag_pole);
    bitptr_memclr(diag2, delka_diag_pole);
    bitptr_memclr(sloupce_obsazeno, hrana_sachovnice);

    //Posuvné ukazatele
    bitptr_t sloupce_obsazeno_;
    bitptr_t diag1_;
    bitptr_t diag2_;

    signed char vrat = dfs_backtrack_optimized_core_row_0(queens, pocet_prozk_stavu, start, stop, sloupce_obsazeno_, sloupce_obsazeno, diag1_, diag1, diag2_, diag2, sloupce_obsazeno_bitptr_v, diag1_bitptr_v, diag2_bitptr_v);

    end:
    for (int i=0;i < POCET_b_POLI_K_UVOLNENI;i++) {
        if (k_uvolneni[i]->byte) bitptr_free_(*k_uvolneni[i]);
        else break;
    }
    for (int i=0;i < POCET_c_POLI_K_UVOLNENI;i++) {
        if (*k_uvolneni_c[i]) free(*k_uvolneni_c[i]);
        else break;
    }
    return vrat;

    alokacni_chyba_l:
    alokacni_chyba();
    vrat = 1;
    goto end;
}
static inline signed char dfs_backtrack_optimized(signed long *queens, unsigned long *pocet_prozk_stavu) {
    return rozdel_praci(dfs_backtrack_optimized_thread, queens, pocet_prozk_stavu);
}

// -------------------------------------------------------------
// Zachycení signálů
// -------------------------------------------------------------

volatile sig_atomic_t ctrlc = 0;

#ifdef _WIN32
BOOL WINAPI console_handler(DWORD signal) {
    switch (signal) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            ctrlc = 1;
            zastav_vlakna();
            return TRUE; // signal zpracován
        default:
            return FALSE;
    }
}
#else
void handle_sigint(int sig) {
    ctrlc = 1;
    zastav_vlakna();
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

    portable_atomic_init(&stop_vlaken, 0); //Inicializujeme proměnnou pro zastevení všech vláken kromě hlavního.

#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#else
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);
#endif

    delka_diag_pole = hrana_sachovnice * 2 - 1;
    n_div_2_ceil = (hrana_sachovnice>>1) + (hrana_sachovnice&1);
    n_min_1 = hrana_sachovnice -1;
    n_min_2 = hrana_sachovnice -2;

    printf("Porovnání %d algoritmů pro problém %ld královen:\n\n", POCET_ALGORITMU, hrana_sachovnice);

    printf("Probíhá automatické zjišťování počtu fyzických jader...");
    zjisti_pocet_fyzickych_jader();
    if (pocet_fyzickych_jader == 0) {
        printf(" Automatické zjištění selhalo.\n");
        do printf("Zadejte prosím počet fyzických jader ručně: "); while (scanf("%d", &pocet_fyzickych_jader) <= 0 && !ctrlc);
    }
    else printf(" Hotovo.\n");
    if (ctrlc) return 3;

    pocet_sloupcu_na_vlakno = hrana_sachovnice/pocet_fyzickych_jader;
    pocet_sloupcu_na_posl_vlakno = hrana_sachovnice%pocet_fyzickych_jader;

    signed long *queens = malloc(hrana_sachovnice * sizeof(signed long));
    if (!queens) {
        alokacni_chyba();
        return 2;
    }

    char* vetsina_alg_celk_pocet_stavu = vetsina_alg_zjisti_celk_pocet_stavu();
    char* dfs_celk_pocet_stavu_str = dfs_backtrack_zjisti_celk_pocet_stavu();

    printf("Celkový počet stvavů: %s\n", vetsina_alg_celk_pocet_stavu);

    printf("Poznámka: DFS prochází stavový prostor jinak, takže se počítají i stavy, kdy není umístěna královna ve všech sloupcích. Tento program ale počítá procento projitých stavů vzhledem k počtu pro většinu algoritmů.\nCelkový počet stavů pro DFS: %s\n", dfs_celk_pocet_stavu_str);
    alg_info(dfs_backtrack_optimized, queens, "3) DFS s backtrackingem - optimalizováno");

    free(vetsina_alg_celk_pocet_stavu);
    free(dfs_celk_pocet_stavu_str);
    fmpz_clear(celk_pocet_stavu);

    free(queens);

    return 0;
}
