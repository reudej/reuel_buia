#!/usr/bin/env python

# Tento soubor byl automaticky vygenerován pomocí predictor_generator.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

pd.set_option("future.no_silent_downcasting", True)


data = pd.read_csv("weatherAUS.csv")



# Začátek vloženého úseku pro úpravu dat ještě před predikcí.

str_num_list_svet_strany = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]
replacement_dict = {zkratka: i for i, zkratka in enumerate(str_num_list_svet_strany)}
replacement_dict["No"] = 0.0
replacement_dict["Yes"] = 0.1

for i, misto in enumerate(data["Location"].unique().tolist()):
    replacement_dict[misto] = i/10

data = data.replace(replacement_dict)
data = data.fillna(-0.1)

data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").map(pd.Timestamp.toordinal) / 10
data *= 10
data = data.astype(int)


# Konec vloženého úseku




X_all = data[['WindSpeed3pm', 'WindGustSpeed', 'Humidity3pm', 'MaxTemp', 'Evaporation', 'MinTemp', 'Date', 'WindDir3pm', 'WindDir9am', 'RainToday', 'WindGustDir', 'Pressure3pm', 'Sunshine', 'Pressure9am', 'Location', 'Cloud9am', 'Cloud3pm', 'WindSpeed9am', 'Temp3pm', 'Rainfall', 'Temp9am', 'Humidity9am']]
y_all = data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)

print('Velikost trenovaci mnoziny:', len(X_train))
print('Velikost testovaci mnoziny:', len(X_test))

model = RandomForestClassifier(n_estimators=500)
print(dir(model))
model.fit(X_train, y_train)
y_pred_loss = model.predict_proba(X_test)
y_pred_acc = model.predict(X_test)

print ("ACCURACY:", accuracy_score(y_test, y_pred_acc))
print ("LOSS:", log_loss(y_test, y_pred_loss))
