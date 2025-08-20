
import pandas as pd
caps = pd.read_csv("contract_caps_2121.csv")
S = caps.loc[caps.sponsored==True, "cap"].sum()         # total sponsored capacity
U = 188_006                                             # users in this drop
k = 3
unsponsored_needed_min = max(0, U*k - S)                # if we fully use sponsored
unsponsored_cap = 370_000
feasible = unsponsored_needed_min <= unsponsored_cap
print(S, unsponsored_needed_min, feasible)