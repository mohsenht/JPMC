import pandas as pd
import numpy as np

numeric_parameters = [
    'age',
    'wage per hour',
    'capital gains',
    'capital losses',
    'dividends from stocks',
    'num persons worked for employer',
    'weeks worked in year',
    'wage_x_weeks',
]

flags = [
    'wage_is_zero', 'capgain_has', 'capgain_is_max',
    'caploss_has', 'dividends_has', 'worked_any',
    'sex_male', 'year_is_95'
]

one_hot_columns = [
    'class of worker',
    'detailed industry recode',
    'detailed occupation recode',
    'education',
    'enroll in edu inst last wk',
    'marital stat',
    'major industry code',
    'major occupation code',
    'race',
    'hispanic origin',
    'member of a labor union',
    'reason for unemployment',
    'full or part time employment stat',
    'tax filer stat',
    # 'region of previous residence',
    # 'state of previous residence',
    # 'detailed household and family stat',
    'detailed household summary in household',
    # 'migration code-change in msa',
    # 'migration code-change in reg',
    # 'migration code-move within reg',
    # 'live in this house 1 year ago',
    'migration prev res in sunbelt',
    'family members under 18',
    # 'country of birth father',
    # 'country of birth mother',
    # 'country of birth self',
    'citizenship',
    'own business or self employed',
    "fill inc questionnaire for veteran's admin",
    'veterans benefits',
]

def build_feature_vector(df, columns):
    df['wage_x_weeks'] = np.log1p(
        df['wage per hour'].fillna(0).clip(lower=0) *
        df['weeks worked in year'].fillna(0).clip(lower=0)
    )

    for col in numeric_parameters:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['wage_is_zero'] = (df['wage per hour'] == 0).astype(int)
    df['capgain_has'] = (df['capital gains'] > 0).astype(int)
    df['capgain_is_max'] = (df['capital gains'] == 99999).astype(int)
    df['caploss_has'] = (df['capital losses'] > 0).astype(int)
    df['dividends_has'] = (df['dividends from stocks'] > 0).astype(int)
    df['worked_any'] = (df['weeks worked in year'] > 0).astype(int)
    df['sex_male'] = (df['sex'] == 'Male').astype(int)
    df['year_is_95'] = (df['year'] == 95).astype(int)

    for col in one_hot_columns:
        if col in df.columns:
            df[col] = df[col].astype('object')

    X_num = df[numeric_parameters + flags].copy()
    X_cat = pd.get_dummies(df[one_hot_columns], prefix=one_hot_columns, prefix_sep='=', drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    print(f"Final feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns")

    return X, columns
