#!/usr/bin/python3

nadoby = [4, 3]
poc_stav = [0, 0]
cil_stavy = [[0,2], [2,0], [1,1]]
zname_stavy = [poc_stav.copy()]
strom = [poc_stav, None, None, None]

def prozkoumej(podstrom, stav, nadoba):
    for akce in range(1,4):
        akt_stav = stav.copy()
        if akce == 1:
            akt_stav[nadoba] = nadoby[nadoba]
        elif akce == 2:
            akt_stav[nadoba] = 0
        elif akce == 3:
            druha_nadoba = int(not nadoba)
            odecist = akt_stav[nadoba] + akt_stav[druha_nadoba] - nadoby[druha_nadoba]
            if odecist < 0: odecist = 0
            akt_stav[druha_nadoba] += akt_stav[nadoba]
            if akt_stav[druha_nadoba] >= nadoby[druha_nadoba]: akt_stav[druha_nadoba] = nadoby[druha_nadoba]
            akt_stav[nadoba] -= odecist
            if akt_stav[nadoba] < 0: akt_stav[nadoba] = 0
        podstrom[akce] = [akt_stav, None, None, None]
        
        if not akt_stav in zname_stavy:
            prozkoumej(podstrom[akce], akt_stav)
