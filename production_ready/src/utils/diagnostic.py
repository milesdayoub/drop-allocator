import pandas as pd, sys, collections, numpy as np, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

elig = pd.read_csv("elig_pairs_2121.csv")
caps = pd.read_csv("contract_caps_2121.csv")
# eligible counts per user
cnt = elig.groupby('user_id').size()
logger.info("eligible rows ≤1 : %s", (cnt<=1).sum())
logger.info("eligible rows  =2 : %s", (cnt==2).sum())
logger.info("eligible rows  =3 : %s", (cnt==3).sum())
logger.info("eligible rows >=4 : %s", (cnt>=4).sum())
# how many of those low‑elig users have only unsponsored contracts?
unsponsored = set(caps.loc[caps.is_sponsored==False, 'contract_address'])
def only_unsponsored(u):
    rows = elig.loc[elig.user_id==u, 'contract_address']
    return all(c in unsponsored for c in rows)
logger.info("≤2‑elig users with only unsponsored rows: %s", 
            sum(only_unsponsored(u) for u,c in cnt[cnt<=2].items()))