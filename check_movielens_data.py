import pandas as pd
from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict


ex = Experiment()
ex = initialise(ex)

@ex.automain
def main(_run):
    args = edict(_run.config)
    raw_data = pd.read_csv("./ml-latest-small/ratings.csv")

    raw_data.info()
    data = pd.DataFrame({
        "request": raw_data["movieId"],
        "timestamp": raw_data["timestamp"]
    })
    data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]
    max_val_new = int(data['request'].max() / args.squeeze_factor)
    data["request"] = data["request"] % max_val_new
    data = data.sort_values(by=["timestamp"])

    data = data.drop_duplicates()
    data.info()
    print(data["request"].max())
    data.to_csv('./ml-latest-small/movielens_cleaned.csv', index=False)
