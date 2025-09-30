from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from src.feature_selection import build_feature_vector, numeric_parameters, flags, one_hot_columns
from src.utility import load_data, clean_column_name

df, columns = load_data()
columns = clean_column_name(columns)
X, columns = build_feature_vector(df, columns)

num_cols = numeric_parameters + flags

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=False), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), one_hot_columns),
    ],
    remainder='drop',
    sparse_threshold=0.3
)

embed = TruncatedSVD(n_components=40, random_state=89)

X = df[num_cols + one_hot_columns]
w = df['weight'].values

def pick_cluster_number():
    scores = []
    for K in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        pipe_k = Pipeline([
            ('prep', preprocess),
            ('svd', embed),
            ('km', KMeans(n_clusters=K, n_init='auto', random_state=89))
        ])
        pipe_k.fit(X, km__sample_weight=w)

        preprocessed_X = pipe_k.named_steps['prep'].transform(X)
        svd_X = pipe_k.named_steps['svd'].transform(preprocessed_X)

        sil = silhouette_score(svd_X, pipe_k.named_steps['km'].labels_, metric='euclidean')
        scores.append((K, sil))

    print(scores)


pipe = Pipeline([
    ('prep', preprocess),
    ('svd', embed),
    ('km', KMeans(n_clusters=7, n_init='auto', random_state=89))
])

pipe.fit(X, km__sample_weight=w)

df['cluster'] = pipe.named_steps['km'].labels_
