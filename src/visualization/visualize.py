from src.models.predict_models import load_models
from src.models.common import load_data, rate_model
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    _, df = load_data()
    cb_multi, cbs, sgds = load_models()
    X, y = df.iloc[:, :-5], df.iloc[:, -5:]
    y_pred = zip(*cb_multi.predict(X))
    y_test = [col for name, col in y.items()]
    print(rate_model(zip(y_test, y_pred))[0])


if __name__ == "__main__":
    main()
