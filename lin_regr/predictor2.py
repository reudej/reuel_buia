#!/usr/bin/env python

# Tento soubor byl automaticky vygenerován pomocí predictor_generator.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

data = pd.read_csv("forestfires.csv")



# Začátek vloženého úseku pro úpravu dat ještě před predikcí.

data["month_num"] = pd.to_datetime(data["month"], format="%b").dt.month



# Konec vloženého úseku




X_all = data[['wind', 'month_num', 'FFMC', 'RH', 'ISI', 'X', 'rain', 'DMC', 'DC', 'temp', 'Y']]
y_all = data['area']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)
y_train **= 0.1

print('Velikost trenovaci mnoziny:', len(X_train))
print('Velikost testovaci mnoziny:', len(X_test))

model = RandomForestRegressor(n_estimators=157)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) ** 10

print ("MAE:", mean_absolute_error(y_test, y_pred))
print ("RMSE:", root_mean_squared_error(y_test, y_pred))
