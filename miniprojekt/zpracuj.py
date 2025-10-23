#!/usr/bin/python3
import pandas as pd
odchylka = float(input("Zadejte prosím max. odchylku: "))
df = pd.read_csv("csv.csv", index_col=0)
komodity = df.columns.to_list()
for i in range(len(komodity)):
    
