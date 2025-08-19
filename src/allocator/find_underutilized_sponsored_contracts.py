# scripts/find_underutilized_sponsored_contracts.py
import pandas as pd

CAPS = "data/contract_caps_2122_v3.csv"
ASSN = "outputs/assignments.csv"

caps = pd.read_csv(CAPS, dtype={'contract_address': str})
assn = pd.read_csv(ASSN, dtype={'contract_address': str})

# only sponsored
caps_s = caps[caps['is_sponsored'] == True][['contract_address','cap_face']].copy()

# assigned counts per contract from the last run
assigned = (assn.groupby('contract_address')
                 .size()
                 .rename('assigned')
                 .reset_index())

m = caps_s.merge(assigned, on='contract_address', how='left').fillna({'assigned': 0})
m['util'] = m['assigned'] / m['cap_face']
m['remaining'] = m['cap_face'] - m['assigned']

# Pick the 14 with the most remaining headroom (you can alternatively threshold on util < 0.60)
under = (m[m['remaining'] > 0]
         .sort_values(['remaining','util'], ascending=[False, True])
         .head(14))

# 1) Save a simple text list (one per line)
under['contract_address'].to_csv("outputs/underutilized_sponsored_2122.txt", index=False, header=False)

# 2) Print a ClickHouse array literal you can paste
arr = "arrayDistinct([" + ",".join(f"'{a}'" for a in under['contract_address']) + "])"
print(arr)

# 3) Also show a quick table for sanity
print(under[['contract_address','cap_face','assigned','remaining','util']])
