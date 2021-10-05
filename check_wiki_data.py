import pandas as pd
from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict


ex = Experiment()
ex = initialise(ex)

@ex.automain
def main(_run):
    args = edict(_run.config)
    with open('./data/CMU_truncated_2500files_116k_requests.txt', 'r') as f:
        raw_data = f.readlines()
    raw_data = [x.strip().split() for x in raw_data]
    fileid = [int(x[1]) for x in raw_data]
    timestamp = [int(x[0]) for x in raw_data]

    print(len(raw_data))
    import pdb; pdb.set_trace()
    # raw_data = pd.read_csv("./ml-latest-small/ratings.csv")

    data = pd.DataFrame({
        "request": fileid,
        "timestamp": timestamp,
    })
    data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]
    max_val_new = int(data['request'].max() / args.squeeze_factor)
    data["request"] = data["request"] % max_val_new
    data = data.sort_values(by=["timestamp"])

    data = data.drop_duplicates()
    data.info()
    print(data["request"].max())
    import pdb; pdb.set_trace()
    data.to_csv('./data/wiki_cleaned.csv', index=False)
    # data.to_csv('./ml-latest-small/movielens_cleaned.csv', index=False)
