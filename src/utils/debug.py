import pandas as pd
df = pd.read_csv("elig_pairs_2121_top30_strat_v2.csv")
print("raw_users:", df['user_id'].nunique())
print("raw_pairs:", len(df))


df = pd.read_csv("elig_pairs_2121_top30_strat_v2.csv")
df = df[df['score'] >= 1e-9]
df = df.sort_values(['user_id','score'], ascending=[True, False])  # important
df = df.groupby('user_id').head(15)  # match --top_n 15
print("users_after_filters:", df['user_id'].nunique())
print("pairs_after_filters:", len(df))

df = pd.read_csv("elig_pairs_2121_top30_strat_v2.csv")
print("users with max score >= 1e-9:", (df.groupby('user_id')['score'].max() >= 1e-9).sum())
print("users with max score == 0:", (df.groupby('user_id')['score'].max() == 0).sum())