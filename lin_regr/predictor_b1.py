#!/usr/bin/env python

# Tento soubor byl automaticky vygenerován pomocí predictor_generator.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

data = pd.read_csv("https://raw.githubusercontent.com/mlcollege/ai-academy/main/07-klasifikace/data/infarkt.csv")



# Začátek vloženého úseku pro úpravu dat ještě před predikcí.



# Konec vloženého úseku




X_all = data[['vek', 'manzelstvi', 'id', 'bmi', 'pohlavi', 'bydliste', 'zamestnani', 'nemoc_srdce', 'cukr', 'hypertenze', 'koureni']]
y_all = data['infarkt']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)

print('Velikost trenovaci mnoziny:', len(X_train))
print('Velikost testovaci mnoziny:', len(X_test))

model = RandomForestRegressor(n_estimators=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("MAE:", mean_absolute_error(y_test, y_pred))
print ("RMSE:", root_mean_squared_error(y_test, y_pred))
