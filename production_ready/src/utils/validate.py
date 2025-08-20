#!/usr/bin/env python3
import argparse, sys
import pandas as pd
import numpy as np

def fmt(n): 
    return f"{int(n):,}" if pd.notna(n) else "n/a"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--elig", required=True)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--unsponsored_cap", type=int, default=0, help="global cap for unsponsored assignments (allocator flag)")
    args = ap.parse_args()

    # Load caps (drop list for this run)
    caps = pd.read_csv(args.caps, dtype={'contract_address': str})
    need_caps = {'contract_address','is_sponsored','cap_face'}
    missing = need_caps - set(caps.columns)
    if missing:
        raise ValueError(f"caps missing columns: {missing}")
    caps['is_sponsored'] = caps['is_sponsored'].astype(bool)
    caps = caps.drop_duplicates('contract_address')

    # Load elig pairs (restricted to this drop)
    e = pd.read_csv(args.elig, dtype={'user_id': str, 'contract_address': str})
    need_e = {'user_id','contract_address','score'}
    missing = need_e - set(e.columns)
    if missing:
        raise ValueError(f"elig missing columns: {missing}")
    # Tag sponsored/unsponsored; keep only contracts that are in caps
    e = e.merge(caps[['contract_address','is_sponsored']], on='contract_address', how='inner', validate='many_to_one')

    # Quick coverage view: do users even have >=k options?
    per_user_opts = e.groupby('user_id').size().rename('opt_total')
    u_all = per_user_opts.index.nunique()
    users_with_ge_k_opts = int((per_user_opts >= args.k).sum())
    users_with_lt_k_opts = u_all - users_with_ge_k_opts

    # Split sponsored vs unsponsored options (helps diagnose shortages)
    e_counts = e.pivot_table(index='user_id', columns='is_sponsored', values='contract_address', aggfunc='nunique', fill_value=0)
    e_counts.columns = ['unsponsored_opts' if c==False else 'sponsored_opts' for c in e_counts.columns]
    e_counts = e_counts.reindex(per_user_opts.index).fillna(0).astype(int)
    e_counts['opt_total'] = per_user_opts
    lacking = e_counts[e_counts['opt_total'] < args.k]

    # Theoretical feasibility vs. capacity
    S_cap = int(caps.loc[caps.is_sponsored, 'cap_face'].fillna(0).sum())
    U = u_all
    total_needed = U * args.k
    min_unspon_needed = max(0, total_needed - S_cap)
    feasible_by_caps = (args.unsponsored_cap >= min_unspon_needed)

    # Load assignments
    a = pd.read_csv(args.assign, dtype={'user_id': str, 'contract_address': str})
    need_a = {'user_id','contract_address','score'}
    missing = need_a - set(a.columns)
    if missing:
        raise ValueError(f"assign missing columns: {missing}")
    a = a.merge(caps[['contract_address','is_sponsored','cap_face']], on='contract_address', how='left', validate='many_to_one')
    # Flag any assignments to contracts not in caps (should be zero)
    not_in_caps = a['cap_face'].isna().sum()
    a = a.dropna(subset=['cap_face'])
    a['cap_face'] = a['cap_face'].astype(int)

    # Assignment distribution per user
    per_user_assigned = a.groupby('user_id').size()
    n3 = int((per_user_assigned == args.k).sum())
    n2 = int((per_user_assigned == args.k-1).sum())
    n1 = int((per_user_assigned == 1).sum())
    n0 = int(U - per_user_assigned.index.nunique())

    # Sponsored vs unsponsored realized
    total_assigned = len(a)
    assigned_spon = int(a['is_sponsored'].sum())
    assigned_unspon = total_assigned - assigned_spon

    # Capacity checks
    per_c = a.groupby(['contract_address','is_sponsored']).size().rename('assigned').reset_index()
    per_c = per_c.merge(caps[['contract_address','is_sponsored','cap_face']], on=['contract_address','is_sponsored'], how='left')
    per_c['over_by'] = per_c['assigned'] - per_c['cap_face']
    overages = per_c[(per_c['is_sponsored']==True) & (per_c['over_by'] > 0)]

    # Global unsponsored cap check
    unspon_cap_ok = (assigned_unspon <= args.unsponsored_cap) if args.unsponsored_cap > 0 else True

    # Print summary
    print("── validator ─────────────────────────────────────────")
    print(f"eligible users (≥1 option)     : {fmt(U)}")
    print(f"users with ≥{args.k} options     : {fmt(users_with_ge_k_opts)}")
    print(f"users with <{args.k} options     : {fmt(users_with_lt_k_opts)}")
    print()
    print("Theoretical feasibility (before solving):")
    print(f"  sponsored cap sum (Σ cap_face, sponsored) : {fmt(S_cap)}")
    print(f"  global unsponsored_cap (flag)             : {fmt(args.unsponsored_cap)}")
    print(f"  total slots needed (U×k)                  : {fmt(total_needed)}")
    print(f"  min unsponsored needed (U×k - S_cap)      : {fmt(min_unspon_needed)}")
    print(f"  feasible by caps?                         : {'YES' if feasible_by_caps else 'NO'}")
    if not feasible_by_caps:
        print("  → Impossible to give every eligible user k offers with current sponsored caps and unsponsored_cap.")
    if users_with_lt_k_opts:
        print(f"  → {fmt(users_with_lt_k_opts)} users lack enough candidate contracts in the elig file (increase pair coverage).")
    print()
    print("Realized assignments (from allocator output):")
    print(f"  total assignments              : {fmt(total_assigned)}")
    print(f"  users with 3 / 2 / 1 / 0       : {fmt(n3)} / {fmt(n2)} / {fmt(n1)} / {fmt(n0)}")
    print(f"  sponsored / unsponsored        : {fmt(assigned_spon)} / {fmt(assigned_unspon)}")
    print(f"  unsponsored_cap respected?     : {'YES' if unspon_cap_ok else 'NO'}")
    if not_in_caps:
        print(f"  WARN: {fmt(not_in_caps)} assignment rows reference contracts not in caps (dropped in checks).")
    if not overages.empty:
        print("\nOver-cap sponsored contracts:")
        for _, r in overages.sort_values('over_by', ascending=False).iterrows():
            print(f"  {r.contract_address}  assigned={fmt(r.assigned)}  cap={fmt(r.cap_face)}  over_by={fmt(r.over_by)}")
    print("───────────────────────────────────────────────────────")

    # Optional: top reasons users lack k options
    if users_with_lt_k_opts:
        lacking['need'] = args.k - lacking['opt_total']
        buckets = lacking[['sponsored_opts','unsponsored_opts','need']].copy()
        print("\nUsers lacking options (sample breakdown):")
        print(buckets.head(10).to_string(index=False))

if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    main()
