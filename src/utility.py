import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_selection import build_feature_vector

def sanitize(name: str) -> str:
    s = str(name)
    s = s.replace('<', 'lt').replace('>', 'gt')
    s = s.replace('[', '(').replace(']', ')')
    s = s.replace(' ', '_').replace('/', '_').replace('&', 'and')
    s = re.sub(r'[^0-9a-zA-Z_:=().-]', '_', s)
    return s


def clean_column_name(columns):
    seen = {}
    new_cols = []
    for c in columns:
        base = sanitize(c)
        if base in seen:
            seen[base] += 1
            base = f"{base}__{seen[base]}"
        else:
            seen[base] = 0
        new_cols.append(base)

    return new_cols


def load_data():
    data_file = "census-bureau.data"
    columns_file = "census-bureau.columns"

    with open(columns_file, "r") as f:
        columns = [line.strip() for line in f.readlines()]

    df = pd.read_csv(data_file, names=columns)

    return df, columns


def train_test_weight_split():
    global columns, X_train, X_test, y_train, y_test, w_train, w_test
    df, columns = load_data()
    columns = clean_column_name(columns)
    X, columns = build_feature_vector(df, columns)
    df['y'] = (df['label'] == '50000+.').astype(int)
    y = df['y'].values
    weights = df['weight'].astype(float).values
    return train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=89,
        stratify=y
    )