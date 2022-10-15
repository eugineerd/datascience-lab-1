import pandas as pd
import cmath


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("ID")
    df["Пол"] = df["Пол"].fillna(df["Пол"].mode()[0])
    df = df.drop("ID_y", axis=1)
    df = process_smoking(df)
    df = process_passive_smoking(df)
    df = process_sleeping(df)
    df = process_alcho(df)
    return df


def process_alcho(df: pd.DataFrame) -> pd.DataFrame:
    rating_map = {
        "никогда не употреблял": 0.0,
        "ранее употреблял": 0.5,
        "употребляю в настоящее время": 1.0,
    }
    df["Алкоголь"] = df["Алкоголь"].map(rating_map)
    df["Возраст алког"] = df["Возраст алког"].fillna(df["Возраст алког"].mean())

    return df


def process_smoking(df: pd.DataFrame) -> pd.DataFrame:
    rating_map = {
        "Никогда не курил(а)": 0.0,
        "Бросил(а)": 0.5,
        "Курит": 1.0,
    }
    df["Статус Курения"] = df["Статус Курения"].map(rating_map)
    df["Возраст курения"] = df["Возраст курения"].fillna(df["Возраст курения"].mean())

    mean_cigs = df["Сигарет в день"].mean()

    def replace_cigs_num(row):
        if not cmath.isnan(row["Сигарет в день"]):
            val = row["Сигарет в день"]
        elif row["Статус Курения"] == "Курит":
            val = mean_cigs
        elif row["Статус Курения"] == "Бросил(а)":
            val = mean_cigs * 0.5
        else:
            val = 0.0
        row["Сигарет в день"] = val
        return row

    df = df.apply(lambda row: replace_cigs_num(row), axis=1)
    return df


def process_passive_smoking(df: pd.DataFrame) -> pd.DataFrame:
    pass_smoking_map = {
        "1-2 раза в неделю": 1.0 / 5.0,
        "3-6 раз в неделю": 2.0 / 5.0,
        "не менее 1 раза в день": 3.0 / 5.0,
        "2-3 раза в день": 4.0 / 5.0,
        "4 и более раз в день": 5.0 / 5.0,
        pd.NA: pd.NA,
    }
    df["Частота пасс кур"] = df["Частота пасс кур"].map(pass_smoking_map)
    pass_smoking_mean = df["Частота пасс кур"].mean()

    def replace_pass_smoking(row):
        if not cmath.isnan(row["Частота пасс кур"]):
            val = row["Частота пасс кур"]
        elif row["Пассивное курение"] == 1:
            val = pass_smoking_mean
        else:
            val = 0.0
        row["Частота пасс кур"] = val
        return row

    df = df.apply(lambda row: replace_pass_smoking(row), axis=1)
    return df


def process_sleeping(df: pd.DataFrame) -> pd.DataFrame:
    def time_to_hours(x):
        h, s, m = map(int, x.split(":"))
        return h + m / 60 + s / 60 / 60

    start_times = df["Время засыпания"].map(time_to_hours)
    # Для расположения времени до полночи и после рядом
    df["Время засыпания"] = (start_times + 11) % 24
    end_times = df["Время пробуждения"].map(time_to_hours)
    df["Время пробуждения"] = end_times
    sleep_duration = ((end_times - start_times) % 24).rename("Продолжительность сна")
    df = pd.concat([df.iloc[:, :-5], sleep_duration, df.iloc[:, -5:]], axis=1)
    return df
