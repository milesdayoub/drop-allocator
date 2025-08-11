#!/usr/bin/env python3
import argparse, sys
import pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--elig", required=True)   # needed to compute 0-offer users
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--top_constrained", type=int, default=15)
    args = ap.parse_args()

    caps = pd.read_csv(args.caps, dtype={'contract_address': str})
    need_caps = {'contract_address','cap_face','is_sponsored'}
    if (need_caps - set(caps.columns)):
        raise ValueError(f"caps missing {need_caps - set(caps.columns)}")
    caps['is_sponsored'] = caps['is_sponsored'].astype(bool)
    caps = caps.drop_duplicates('contract_address')

    a = pd.read_csv(args.assign, dtype={'contract_address': str, 'user_id': str})
    need_a = {'user_id','contract_address','score'}
    if (need_a - set(a.columns)):
        raise ValueError(f"assign missing {need_a - set(a.columns)}")

    elig_users = pd.read_csv(args.elig, usecols=['user_id'], dtype={'user_id': str}).user_id.dropna().drop_duplicates()

    # join caps to tag sponsored
    a = a.merge(caps[['contract_address','is_sponsored','cap_face']],
                on='contract_address', how='left', validate='many_to_one')
    a = a.dropna(subset=['cap_face'])  # drop rows not in caps

    # 3/2/1/0 breakdown
    per_user = a.groupby('user_id').size()
    n3 = int((per_user == args.k).sum())
    n2 = int((per_user == args.k-1).sum())
    n1 = int((per_user == 1).sum())
    n0 = int(elig_users.nunique() - per_user.index.nunique())

    # sponsored share
    total = int(len(a))
    s_assign = int(a['is_sponsored'].sum())
    s_share = (s_assign / total * 100) if total else 0.0

    # top constrained contracts (at cap or ≥95% util)
    per_c = a.groupby(['contract_address','is_sponsored']).size().rename('assigned').reset_index()
    per_c = per_c.merge(caps[['contract_address','is_sponsored','cap_face']],
                        on=['contract_address','is_sponsored'], how='left')
    per_c['util'] = per_c['assigned'] / per_c['cap_face'].replace(0, np.nan)
    constrained = per_c[(per_c['cap_face'] > 0) & ((per_c['assigned'] >= per_c['cap_face']) | (per_c['util'] >= 0.95))]
    constrained = constrained.sort_values(['util','cap_face','assigned'], ascending=[False, False, False]).head(args.top_constrained)

    print("── report ─────────────────────────────")
    print(f"assignments       : {total:,}")
    print(f"eligible users    : {elig_users.nunique():,}")
    print(f"offer counts      : 3→{n3:,}   2→{n2:,}   1→{n1:,}   0→{n0:,}")
    print(f"sponsored share   : {s_share:.2f}%  ({s_assign:,}/{total:,})")
    if not constrained.empty:
        print("\nTop constrained contracts:")
        for _, r in constrained.iterrows():
            util = "n/a" if pd.isna(r['util']) else f"{r['util']*100:.1f}%"
            print(f"  {r.contract_address}  sponsored={'Y' if r.is_sponsored else 'N'}  "
                  f"assigned={int(r.assigned):>6}  cap={int(r.cap_face):>6}  util={util}")
    print("──────────────────────────────────────")

if __name__ == "__main__":
    main()
