import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import recall_score
from typing import List, Tuple
from sklearn.model_selection import train_test_split

CAT_FEATURES = []


def get_cat_features(df: pd.DataFrame) -> List[str]:
    return list(df.columns[df.dtypes == "object"])


# Почему такая метрика?
# Нам интересн recall болезней для уменьшения ошибки первого рода
def rate_model(y_pairs) -> Tuple[float, List[float]]:
    scores = []
    for y_true, y_pred in y_pairs:
        scores.append(recall_score(y_true, y_pred))
    return np.mean(scores), scores  # type: ignore


def default_train_split(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    return X_train, X_test, y_train, y_test  # type: ignore


def load_data():
    with open("data/processed/train.pkl", "rb") as f:
        df = pickle.load(f)
    num = int(0.8 * len(df))
    return df.iloc[:num], df.iloc[num:]
