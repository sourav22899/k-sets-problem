import pandas as pd
import numpy as np

# raw_data = pd.read_csv("./mit/out.mit", sep="\t")
# raw_data = raw_data.reset_index(level=[0, 1])
# raw_data = pd.DataFrame({
#     "P1/P2": raw_data["level_0"],
#     "weight": raw_data["level_1"],
#     "timestamp": raw_data["% sym positive"]
# })
#
# person = list(raw_data["P1/P2"].astype(str))
# person1 = [int(x.split(' ')[0]) for x in person]
# person2 = [int(x.split(' ')[1]) for x in person]
#
# raw_data = pd.DataFrame({
#     "P1": person1,
#     "P2": person2,
#     "timestamp": raw_data["timestamp"]
# })
#
# data = raw_data.sort_values(by=["timestamp"])
# data.values.sort(1)
#
# print(data.head(10))
# print(data.count(), len(data["timestamp"].unique()))
# data = data.drop_duplicates()
# data.to_csv('./mit/mit_cleaned.csv', index=False)

data = pd.read_csv("./data/mit_cleaned.csv")
p1 = data["P1"]
p2 = data["P2"]
hash = 100 * np.maximum(p1, p2) + np.minimum(p1, p2)
data["request"] = hash
data = data.drop_duplicates(subset=['timestamp', 'request'])
print(data.head())
data.info()
data.to_csv('./data/mit_cleaned.csv', index=False)