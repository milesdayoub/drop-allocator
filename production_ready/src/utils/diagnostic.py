import pandas as pd, sys, collections, numpy as np
elig = pd.read_csv("elig_pairs_2121.csv")
caps = pd.read_csv("contract_caps_2121.csv")
# eligible counts per user
cnt = elig.groupby('user_id').size()
print("eligible rows ≤1 :", (cnt<=1).sum())
print("eligible rows  =2 :", (cnt==2).sum())
print("eligible rows  =3 :", (cnt==3).sum())
print("eligible rows >=4 :", (cnt>=4).sum())
# how many of those low‑elig users have only unsponsored contracts?
unsponsored = set(caps.loc[caps.is_sponsored==False, 'contract_address'])
def only_unsponsored(u):
    rows = elig.loc[elig.user_id==u, 'contract_address']
    return all(c in unsponsored for c in rows)
print("≤2‑elig users with only unsponsored rows:", 
      sum(only_unsponsored(u) for u,c in cnt[cnt<=2].items()))