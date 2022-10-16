import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.models.common import get_cat_features
from typing import Dict, List


def make_sgd_one_for_each_category(
    X: pd.DataFrame, y: pd.DataFrame, custom_params: Dict = dict()
) -> List[Pipeline]:
    cat_features = get_cat_features(X)
    clfs = []
    params = dict(
        verbose=0,
        random_state=1,
        class_weight="balanced",
        loss="log_loss",
        alpha=0.0001,
        validation_fraction=0.2,
    )
    params.update(custom_params)

    for col in y.columns:
        col_tr = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(), cat_features)]
        )
        pipe = Pipeline(
            steps=[
                ("col_tr", col_tr),
                ("clf", SGDClassifier(**params)),
            ]
        )
        pipe.fit(X, y[col])
        clfs.append(pipe)
    return clfs
