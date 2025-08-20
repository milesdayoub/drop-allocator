import pandas as pd, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
df = pd.read_csv("elig_pairs_2121_top30_strat_v2.csv")
logger.info("raw_users: %s", df['user_id'].nunique())
logger.info("raw_pairs: %s", len(df))


df = pd.read_csv("elig_pairs_2121_top30_strat_v2.csv")
df = df[df['score'] >= 1e-9]
df = df.sort_values(['user_id','score'], ascending=[True, False])  # important
df = df.groupby('user_id').head(15)  # match --top_n 15
logger.info("users_after_filters: %s", df['user_id'].nunique())
logger.info("pairs_after_filters: %s", len(df))

df = pd.read_csv("elig_pairs_2121_top30_strat_v2.csv")
logger.info("users with max score >= 1e-9: %s", (df.groupby('user_id')['score'].max() >= 1e-9).sum())
logger.info("users with max score == 0: %s", (df.groupby('user_id')['score'].max() == 0).sum())