import pickle
from typing import List, Tuple
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline


def load_models() -> Tuple[
    CatBoostClassifier, List[CatBoostClassifier], List[Pipeline]
]:
    with open("models/catboost_multi.pkl", "rb") as f:
        cb: CatBoostClassifier = pickle.load(f)

    cbs: List[CatBoostClassifier] = []
    for i in range(5):
        with open(f"models/catboost_{i}.pkl", "rb") as f:
            m = pickle.load(f)
            cbs.append(m)

    sgds: List[Pipeline] = []
    for i in range(5):
        with open(f"models/sgdclassifier_{i}.pkl", "rb") as f:
            m = pickle.load(f)
            sgds.append(m)

    return cb, cbs, sgds
