import pandas as pd

DATA_PATH = "data/patients.csv"

def load_raw():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess(df):
    # copy and basic encoding
    df2 = df.copy()
    # encode categorical to integers for ML
    df2["urgency_cat"] = df2["urgency"].map({"low":0, "medium":1, "high":2})
    df2["complexity_cat"] = df2["complexity"].map({"low":0, "medium":1, "high":2})
    return df2
