import catboost
import pandas as pd
from src.models.common import get_cat_features, default_train_split
from typing import Dict, List


def make_catboost_multilabel(
    X: pd.DataFrame, y: pd.DataFrame, custom_params: Dict = dict()
) -> catboost.CatBoostClassifier:
    X_train, X_test, y_train, y_test = default_train_split(X, y)
    cat_features = get_cat_features(X)

    params = dict(
        cat_features=cat_features,
        loss_function="MultiLogloss",
        custom_metric=["Recall"],
        class_names=list(y_test.columns),  # type: ignore
    )
    params.update(custom_params)

    cb = catboost.CatBoostClassifier(**params)
    cb.fit(X=X_train, y=y_train, eval_set=(X_test, y_test), use_best_model=True)
    return cb


def make_catboost_one_for_each_category(
    X: pd.DataFrame, y: pd.DataFrame, custom_params: Dict = dict()
) -> List[catboost.CatBoostClassifier]:
    X_train, X_test, y_train, y_test = default_train_split(X, y)
    cat_features = get_cat_features(X)
    clfs = []
    params = dict(
        cat_features=cat_features,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        depth=4,
        learning_rate=0.015,
        # ctr_leaf_count_limit=6,
    )
    params.update(custom_params)
    for col in y_test.columns:
        cb = catboost.CatBoostClassifier(**params)
        cb.fit(
            X=X_train,
            y=y_train[col],  # type: ignore
            eval_set=(X_test, y_test[col]),  # type: ignore
            use_best_model=True,
        )
        clfs.append(cb)
    return clfs
