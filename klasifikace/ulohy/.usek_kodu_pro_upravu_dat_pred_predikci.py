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
