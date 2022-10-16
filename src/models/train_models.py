import pickle
from src.models import catboost_models, sgdclassifier_model
from src.models.common import load_data


def save_models():
    df, _ = load_data()
    X, y = df.iloc[:, :-5], df.iloc[:, -5:]
    cb = catboost_models.make_catboost_multilabel(X, y)
    with open("models/catboost_multi.pkl", "wb") as f:
        pickle.dump(cb, f)

    cbs = catboost_models.make_catboost_one_for_each_category(X, y)
    for i in range(len(cbs)):
        with open(f"models/catboost_{i}.pkl", "wb") as f:
            pickle.dump(cbs[i], f)

    sgds = sgdclassifier_model.make_sgd_one_for_each_category(X, y)
    for i in range(len(sgds)):
        with open(f"models/sgdclassifier_{i}.pkl", "wb") as f:
            pickle.dump(sgds[i], f)


if __name__ == "__main__":
    save_models()
