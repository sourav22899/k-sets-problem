import pandas as pd

with open('./wiki/CMU_truncated_2500files_116k_requests.txt', 'r') as f:
    raw_data = f.readlines()
raw_data = [x.strip().split() for x in raw_data]
fileid = [int(x[1]) for x in raw_data]
timestamp = [int(x[0]) for x in raw_data]

print(len(raw_data))

data = pd.DataFrame({
    "request": fileid,
    "timestamp": timestamp,
})
data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]
data = data.sort_values(by=["timestamp"])
data = data.drop_duplicates()

data.info()
data.to_csv('./wiki_cleaned.csv', index=False)
