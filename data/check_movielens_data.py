import pandas as pd

raw_data = pd.read_csv("./ml-latest-small/ratings.csv")
data = pd.DataFrame({
    "request": raw_data["movieId"],
    "timestamp": raw_data["timestamp"]
})
data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]
data = data.sort_values(by=["timestamp"])
data = data.drop_duplicates()

data.info()
data.to_csv('./movielens_cleaned.csv', index=False)
