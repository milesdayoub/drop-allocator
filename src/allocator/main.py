#!/usr/bin/env python3
"""
Claim – Drop Allocator with coverage encouragement (POC)

• Sponsored caps already include redemption projection (cap_face from PG).
• Unsponsored synthetic cap via --unsponsored_cap (no redemption modeling).

Coverage modes:
  - cov_mode=z          → z_{u,t} coverage vars weighted by --cov_w
  - cov_mode=shortfall  → per-user shortfall penalty (lighter/faster)

Extras:
  - Warm start (greedy -> AddHint on CpModel) with cross-version compatibility.
  - Pre-trim with --top_n and/or --min_score.
  - Keeps Greedy and PuLP variants for comparison.
"""

from __future__ import annotations
import argparse, sys, time, textwrap
import math
import pandas as pd, numpy as np

# Optional solvers
try:
    import pulp
    HAVE_PULP = True
except Exception:
    HAVE_PULP = False

try:
    from ortools.sat.python import cp_model
    HAVE_OR = True
except Exception:
    HAVE_OR = False

MAX_CP_SAT_VARS = 6_000_000  # guard (x vars ≈ len(elig))


def _to_int(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def load_inputs(caps_csv, elig_csv, user_groups_csv, group_ratios_csv, unsponsored_cap, top_n, min_score):
    # Load contract caps
    caps = (pd.read_csv(caps_csv, dtype={'contract_address': str})
              .rename(columns=str.strip))
    if 'is_sponsored' not in caps.columns:
        raise ValueError("caps missing {'is_sponsored'}")
    caps['cap_face'] = _to_int(caps['cap_face'])

    # synthetic capacity for unsponsored rows
    unspon = caps['is_sponsored'].isin([False, 0, '0', 'false', 'False', 'f', 'F'])
    caps.loc[unspon & (caps['cap_face'] <= 0), 'cap_face'] = int(unsponsored_cap)

    caps = (caps[caps.cap_face > 0]
              .drop_duplicates('contract_address')
              .reset_index(drop=True))

    # Load user-group mapping
    user_groups = (pd.read_csv(user_groups_csv, dtype={'user_id': str, 'group_id': str})
                     .rename(columns=str.strip)
                     .drop_duplicates('user_id'))
    
    # Load group sponsorship ratios
    group_ratios = (pd.read_csv(group_ratios_csv, dtype={'group_id': str})
                      .rename(columns=str.strip)
                      .drop_duplicates('group_id'))
    
    # Validate required columns
    need_caps = {'contract_address', 'cap_face', 'is_sponsored'}
    need_user_groups = {'user_id', 'group_id'}
    need_group_ratios = {'group_id', 'sponsorship_ratio'}
    
    if (m := need_caps - set(caps.columns)):
        raise ValueError(f"caps missing {m}")
    if (m := need_user_groups - set(user_groups.columns)):
        raise ValueError(f"user_groups missing {m}")
    if (m := need_group_ratios - set(group_ratios.columns)):
        raise ValueError(f"group_ratios missing {m}")

    # Load eligibility pairs
    elig = (pd.read_csv(elig_csv, dtype={'contract_address': str, 'user_id': str})
              .rename(columns=str.strip)
              .drop_duplicates(['user_id', 'contract_address']))

    need_elig = {'user_id', 'contract_address', 'score'}
    if (m := need_elig - set(elig.columns)):
        raise ValueError(f"elig missing {m}")

    # CRITICAL: Filter eligibility to only users who belong to drop-eligible groups
    print(f"Before user filtering: {len(elig):,} elig pairs, {elig['user_id'].nunique():,} unique users")
    elig = elig[elig.user_id.isin(user_groups.user_id)]
    print(f"After user filtering: {len(elig):,} elig pairs, {elig['user_id'].nunique():,} unique users")
    
    # Keep only contracts with positive cap after synthetic fill
    elig = elig[elig.contract_address.isin(caps.contract_address)]

    # Optional: drop very low/zero scores to shrink model
    if min_score is not None:
        elig = elig[elig['score'] >= float(min_score)]

    # Optional per-user Top-N before solve (0=off)
    if top_n and top_n > 0:
        elig = (elig.sort_values(['user_id', 'score'], ascending=[True, False])
                    .groupby('user_id', sort=False)
                    .head(top_n)
                    .reset_index(drop=True))
    
    # Add contract sponsorship information to eligibility pairs
    caps_sponsorship = caps[['contract_address', 'is_sponsored']].copy()
    elig = elig.merge(caps_sponsorship, on='contract_address', how='left')
    
    # Add group information to eligibility pairs for constraint building
    elig = elig.merge(user_groups[['user_id', 'group_id']], on='user_id', how='left')
    
    # Add sponsorship ratio information
    elig = elig.merge(group_ratios[['group_id', 'sponsorship_ratio']], on='group_id', how='left')
    
    # Validate all users have group and ratio information
    missing_groups = elig[elig['group_id'].isna()]
    if not missing_groups.empty:
        raise ValueError(f"Found {len(missing_groups)} elig pairs with missing group_id")
    
    missing_ratios = elig[elig['sponsorship_ratio'].isna()]
    if not missing_ratios.empty:
        raise ValueError(f"Found {len(missing_ratios)} elig pairs with missing sponsorship_ratio")
    
    return caps.reset_index(drop=True), elig.reset_index(drop=True), user_groups.reset_index(drop=True), group_ratios.reset_index(drop=True)


def summarize(df_assign: pd.DataFrame, elig: pd.DataFrame, k: int):
    users = pd.Index(elig['user_id'].unique())
    per_user = df_assign.groupby('user_id').size().reindex(users, fill_value=0)
    dist = per_user.value_counts().reindex(range(0, k+1), fill_value=0).to_dict()
    fill = per_user.sum() / (len(users) * k) if len(users) and k else 0.0
    return {'n_users': int(len(users)),
            'dist': {int(i): int(dist[i]) for i in dist},
            'fill_rate': float(fill)}


def validate_sponsorship_ratios(df_assign, elig, group_ratios, caps=None):
    if df_assign.empty:
        return
    
    print("\n=== SPONSORSHIP RATIO VALIDATION ===")
    
    # Merge to get group and sponsorship info
    df = df_assign.merge(
        elig[['user_id', 'contract_address', 'group_id', 'sponsorship_ratio', 'is_sponsored']], 
        on=['user_id', 'contract_address'], 
        how='left',
        suffixes=('', '_elig'),
    )

    # 1) Per-user uniformity (no mixed sponsorship)
    per_user = df.groupby('user_id')['is_sponsored'].agg(['nunique', 'first'])
    mixed_users = per_user[per_user['nunique'] > 1]
    if not mixed_users.empty:
        print(f"❌ VIOLATION: {len(mixed_users)} users have mixed sponsorship types!")
        print(mixed_users.head())
    else:
        print("✅ All users have uniform sponsorship types")
    
    # 2) Group-level ratios at USER level (not assignment rows)
    # For each user in each group: mark if they are a sponsored user
    user_flag = (
        df.groupby(['group_id', 'user_id'])['is_sponsored']
          .agg(lambda s: bool(s.max()))   # "any" sponsored => user is sponsored
          .rename('user_is_sponsored')
          .reset_index()
    )

    group_totals = (
        user_flag.groupby('group_id')
                 .agg(total_users=('user_id', 'nunique'),
                      sponsored_users=('user_is_sponsored', 'sum'))
                 .reset_index()
    )

    # Bring in sponsorship_ratio
    ratios = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
    group_stats = group_totals.merge(ratios, on='group_id', how='left')

    group_stats['expected_min_sponsored'] = (group_stats['total_users'] * group_stats['sponsorship_ratio']).apply(math.ceil).astype(int)
    group_stats['actual_ratio'] = group_stats['sponsored_users'] / group_stats['total_users'].clip(lower=1)
    group_stats['ratio_violation'] = group_stats['sponsored_users'] < group_stats['expected_min_sponsored']
    
    violations = group_stats[group_stats['ratio_violation']]
    if not violations.empty:
        print(f"❌ RATIO VIOLATIONS in {len(violations)} groups:")
        print(violations[['total_users', 'sponsored_users', 'expected_min_sponsored', 'sponsorship_ratio', 'actual_ratio']])
    else:
        print("✅ All groups respect sponsorship ratio constraints")
    
    print("\nGroup Summary:")
    print(group_stats[['total_users', 'sponsored_users', 'expected_min_sponsored', 'sponsorship_ratio', 'actual_ratio']].round(3))
    
    # 3) Capacity analysis if caps provided
    if caps is not None:
        analyze_sponsored_capacity(df_assign, caps)


def analyze_sponsored_capacity(df_assign, caps):
    """Analyze sponsored contract capacity utilization"""
    print("\n=== SPONSORSHIP CAPACITY ANALYSIS ===")
    
    # Get contract caps for sponsored contracts
    sponsored_caps = caps[caps['is_sponsored'] == True]
    total_sponsored_cap = sponsored_caps['cap_face'].sum()
    
    # Count sponsored assignments by contract
    sponsored_assignments = df_assign[df_assign['is_sponsored'] == True]
    sponsored_usage = sponsored_assignments.groupby('contract_address').size()
    
    # Merge with caps to see utilization
    capacity_analysis = sponsored_caps.merge(
        sponsored_usage.rename('assigned').reset_index(), 
        on='contract_address', 
        how='left'
    ).fillna(0)
    
    capacity_analysis['utilization_pct'] = (capacity_analysis['assigned'] / capacity_analysis['cap_face'] * 100).round(1)
    capacity_analysis['remaining'] = capacity_analysis['cap_face'] - capacity_analysis['assigned']
    
    print(f"Total sponsored contract capacity: {total_sponsored_cap:,}")
    print(f"Total sponsored assignments made: {len(sponsored_assignments):,}")
    print(f"Overall sponsored capacity utilization: {len(sponsored_assignments)/total_sponsored_cap*100:.1f}%\n")
    
    # Show contracts that are fully utilized
    fully_utilized = capacity_analysis[capacity_analysis['assigned'] >= capacity_analysis['cap_face']]
    if not fully_utilized.empty:
        print(f"🚨 FULLY UTILIZED sponsored contracts ({len(fully_utilized)}):")
        print(fully_utilized[['contract_address', 'cap_face', 'assigned', 'utilization_pct']].head(10))
    
    # Show contracts with remaining capacity
    remaining_capacity = capacity_analysis[capacity_analysis['remaining'] > 0]
    if not remaining_capacity.empty:
        print(f"\n✅ Sponsored contracts with remaining capacity ({len(remaining_capacity)}):")
        print(remaining_capacity[['contract_address', 'cap_face', 'assigned', 'remaining', 'utilization_pct']].head(10))
    
    # Show contracts with no assignments
    unused = capacity_analysis[capacity_analysis['assigned'] == 0]
    if not unused.empty:
        print(f"\n⚠️  Unused sponsored contracts ({len(unused)}):")
        print(unused[['contract_address', 'cap_face']].head(10))
    
    # Summary
    print("\n=== CAPACITY SUMMARY ===")
    print(f"Fully utilized: {len(fully_utilized)} contracts")
    print(f"Partially used: {len(remaining_capacity)} contracts") 
    print(f"Unused: {len(unused)} contracts")
    print(f"Total sponsored contracts: {len(sponsored_caps)}")
    
    if len(fully_utilized) > 0:
        print(f"\n🚨 CAPACITY CONSTRAINT: {len(fully_utilized)} sponsored contracts are at 100% capacity!")
        print("This explains why you can't meet the minimum sponsorship ratios.")
        print("Consider: reducing ratio targets, adding more sponsored contracts, or using soft constraints.")


def print_summary(df_assign, caps, elig, k, label, t_sec):
    used = df_assign.contract_address.value_counts() if not df_assign.empty else pd.Series(dtype=int)
    cap = caps.set_index('contract_address').cap_face.reindex(used.index).fillna(0)
    cap_viol = used[used > cap]
    stats = summarize(df_assign, elig, k)
    obj = df_assign['score'].sum() if 'score' in df_assign.columns and not df_assign.empty else float('nan')
    print(textwrap.dedent(f"""
        ── {label} summary ───────────────────────────────────────
        users (elig)      : {stats['n_users']:,}
        assignments       : {len(df_assign):,}
        Σ score           : {obj:.6f}
        offer counts      : 3→{stats['dist'].get(3,0):,}   2→{stats['dist'].get(2,0):,}   1→{stats['dist'].get(1,0):,}   0→{stats['dist'].get(0,0):,}
        fill rate         : {stats['fill_rate']*100:.2f} %  (of {k} offers per user)
        cap violations    : {'0' if cap_viol.empty else cap_viol.to_dict()}
        wall time         : {t_sec:.1f}s
        ──────────────────────────────────────────────────────────
    """).strip())
    
    # Validate sponsorship ratios if group data is available
    if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
        group_ratios = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        validate_sponsorship_ratios(df_assign, elig, group_ratios, caps)


def greedy(caps, elig, k, seed=42, group_ratios=None, timeout=None):
    """
    FAST brand-focused greedy algorithm that prioritizes business constraints over user scores.
    Uses vectorized operations for 10-100x speed improvement.
    Focuses on: 1) Sponsorship ratios, 2) Brand capacity utilization, 3) User scores
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Pre-process data for fast lookups
    print(f"[greedy] Pre-processing data for {len(elig):,} elig pairs...")
    
    # Convert caps to dict for O(1) lookups
    remaining = caps.set_index('contract_address').cap_face.to_dict()
    
    # Pre-filter and sort sponsored eligibility by score (vectorized)
    sponsored_elig = elig[elig['is_sponsored'] == True].copy()
    sponsored_elig = sponsored_elig.sort_values('score', ascending=False)
    
    # Pre-filter unsponsored eligibility by score (vectorized)
    unsponsored_elig = elig[elig['is_sponsored'] == False].copy()
    unsponsored_elig = unsponsored_elig.sort_values('score', ascending=False)
    
    # Create fast user lookup structures
    user_to_group = elig.groupby('user_id')['group_id'].first().to_dict()
    # Build per-user top-(cache_factor*k) offers dictionaries (more options for top-up)
    cache_factor = 10
    user_to_sponsored_offers = {uid: grp.nlargest(k * cache_factor, 'score')
                                for uid, grp in sponsored_elig.groupby('user_id')}
    user_to_unsponsored_offers = {uid: grp.nlargest(k * cache_factor, 'score')
                                  for uid, grp in unsponsored_elig.groupby('user_id')}
    
    # Track assignments and progress
    chosen = []
    assigned_users = set()
    user_sponsorship = {}  # user_id -> sponsorship_type (True/False)
    
    # Phase 1: Meet sponsorship ratios (vectorized approach)
    if group_ratios is not None:
        print(f"[greedy] Phase 1: Meeting sponsorship ratios for {len(group_ratios)} groups")
        # Reserve time budget: dedicate ~30% of total timeout to Phase 1, leaving the rest for Phase 2
        total_deadline = (start_time + float(timeout)) if timeout else None
        p1_deadline = (start_time + 0.3*float(timeout)) if timeout else None
        
        # Calculate group targets and current progress
        group_targets = {}
        group_current = {}
        
        for _, row in group_ratios.iterrows():
            group_id = row['group_id']
            ratio = row['sponsorship_ratio']
            group_users = elig[elig['group_id'] == group_id]['user_id'].unique()
            group_targets[group_id] = int(math.ceil(len(group_users) * ratio))
            group_current[group_id] = 0
        
        # Sort groups by priority (most behind first)
        groups_by_priority = sorted(
            group_targets.items(),
            key=lambda x: (x[1] - group_current.get(x[0], 0), -x[1]),
            reverse=True
        )
        
        # Process each group to meet targets
        for group_id, target in groups_by_priority:
            if group_current[group_id] >= target:
                continue
                
            # Get users in this group who can receive sponsored offers
            group_users = elig[elig['group_id'] == group_id]['user_id'].unique()
            available_users = [u for u in group_users if u not in assigned_users]
            
            # Sort users by their best sponsored offer score
            user_scores = []
            for user_id in available_users:
                if user_id in user_to_sponsored_offers:
                    try:
                        best_score = user_to_sponsored_offers[user_id]['score'].max()
                        user_scores.append((user_id, best_score))
                    except (KeyError, AttributeError):
                        # Skip users with invalid data
                        continue
            
            # Sort by score and assign
            user_scores.sort(key=lambda x: x[1], reverse=True)
            
            for user_id, _ in user_scores:
                if group_current[group_id] >= target:
                    break
                    
                # Get user's sponsored offers
                if user_id in user_to_sponsored_offers:
                    try:
                        user_offers = user_to_sponsored_offers[user_id]
                        # Try to assign best available sponsored offer
                        for _, offer in user_offers.iterrows():
                            contract = offer['contract_address']
                            if remaining.get(contract, 0) > 0:
                                # Assign this offer
                                chosen.append(offer)
                                remaining[contract] -= 1
                                assigned_users.add(user_id)
                                user_sponsorship[user_id] = True
                                group_current[group_id] += 1
                                break
                    except (KeyError, AttributeError):
                        # Skip users with invalid data
                        continue
                
                # Check timeout (Phase 1 budget)
                if p1_deadline and time.time() > p1_deadline:
                    print(f"[greedy] Phase 1 timeout reached ({timeout}s), stopping early")
                    break
            
            # Check timeout after each group (Phase 1 budget)
            if p1_deadline and time.time() > p1_deadline:
                break
    
    # Phase 2: Top-up all users in rounds to reach k offers, exhausting capacity
    print(f"[greedy] Phase 2: Topping up users to {k} offers in rounds (exhaust capacity)")
    
    # Helpers to track what's already assigned
    assigned_pairs = set()  # (user_id, contract_address)
    user_assigned_count = {}
    for s in chosen:
        try:
            uid = s['user_id']; ca = s['contract_address']
            assigned_pairs.add((uid, ca))
            user_assigned_count[uid] = user_assigned_count.get(uid, 0) + 1
        except Exception:
            continue
    
    all_users_list = elig['user_id'].unique().tolist()
    total_possible = len(all_users_list) * k
    
    def try_assign_offer(offer_row):
        uid = offer_row['user_id']; ca = offer_row['contract_address']
        if (uid, ca) in assigned_pairs:
            return False
        if user_assigned_count.get(uid, 0) >= k:
            return False
        if remaining.get(ca, 0) <= 0:
            return False
        # Assign
        chosen.append(offer_row)
        remaining[ca] -= 1
        assigned_pairs.add((uid, ca))
        user_assigned_count[uid] = user_assigned_count.get(uid, 0) + 1
        return True

    # Sponsored offers accessor (use precomputed top list only; avoid extended scans)
    def get_user_sponsored_offers(user_id):
        return user_to_sponsored_offers.get(user_id, pd.DataFrame())

    # Compute group deficits before Phase 2 for targeting
    deficits_by_group = None
    group_targets_now = {}
    group_current_now = {}
    if group_ratios is not None:
        for _, row in group_ratios.iterrows():
            gid = row['group_id']
            users_in_group = elig[elig['group_id'] == gid]['user_id'].nunique()
            group_targets_now[gid] = int(math.ceil(users_in_group * float(row['sponsorship_ratio'])))
        # current sponsored users with >=1 assignment so far
        for uid, is_spon in user_sponsorship.items():
            if is_spon and user_assigned_count.get(uid, 0) > 0:
                gid = user_to_group.get(uid)
                if gid is not None:
                    group_current_now[gid] = group_current_now.get(gid, 0) + 1
        deficits_by_group = {gid: max(0, group_targets_now.get(gid, 0) - group_current_now.get(gid, 0)) for gid in group_targets_now.keys()}

    # Pre-sweep: seed sponsored assignments for users in deficit groups (one per user), with strict caps
    if deficits_by_group is not None and any(v > 0 for v in deficits_by_group.values()):
        sweep_time_budget = 0.10 * float(timeout) if timeout else None
        sweep_deadline = (start_time + sweep_time_budget) if sweep_time_budget else None
        for gid, deficit in sorted(deficits_by_group.items(), key=lambda x: x[1], reverse=True):
            if deficit <= 0:
                continue
            max_users_for_group = min(int(deficit), 1000)
            processed = 0
            grp_users = elig[elig['group_id'] == gid]['user_id'].unique()
            for uid in grp_users:
                if sweep_deadline and time.time() > sweep_deadline:
                    break
                if total_deadline and time.time() > total_deadline:
                    break
                if processed >= max_users_for_group:
                    break
                if deficits_by_group.get(gid, 0) <= 0:
                    break
                # Skip users locked to unsponsored
                if user_sponsorship.get(uid) is False:
                    continue
                # Try to assign one sponsored offer for this user using top list only
                cand_df = get_user_sponsored_offers(uid)
                if cand_df is None or cand_df.empty:
                    continue
                for _, offer in cand_df.iterrows():
                    ca = offer['contract_address']
                    if (uid, ca) in assigned_pairs:
                        continue
                    if remaining.get(ca, 0) <= 0:
                        continue
                    if try_assign_offer(offer):
                        user_sponsorship[uid] = True
                        if gid is not None:
                            group_current_now[gid] = group_current_now.get(gid, 0) + 1
                            deficits_by_group[gid] = max(0, group_targets_now.get(gid, 0) - group_current_now.get(gid, 0))
                        processed += 1
                        break
                # move to next user regardless
    
    # Round-robin: for t = current_assigned+1 .. k, give at most one additional offer per user per round
    for round_idx in range(1, k+1):
        made_progress = False
        for uid in all_users_list:
            # stop early if we already filled everything or timed out
            if len(chosen) >= total_possible:
                break
            # Phase 2 uses the full remaining time until total deadline
            if total_deadline and time.time() > total_deadline:
                print(f"[greedy] Phase 2 timeout reached ({timeout}s), stopping early")
                break
            # Skip users already at or above this round (e.g., already have >= round_idx offers)
            if user_assigned_count.get(uid, 0) >= round_idx:
                continue
            # Decide/obtain the user's sponsorship type
            if uid not in user_sponsorship:
                # Pick the type that yields the highest next-score assignable offer
                # Force sponsored type for deficit groups if any viable sponsored capacity exists (using top list)
                force_spon = False
                if deficits_by_group is not None:
                    gid = user_to_group.get(uid)
                    if gid is not None and deficits_by_group.get(gid, 0) > 0:
                        check_df = get_user_sponsored_offers(uid)
                        if check_df is not None and not check_df.empty:
                            for _, row in check_df.iterrows():
                                if (uid, row['contract_address']) not in assigned_pairs and remaining.get(row['contract_address'], 0) > 0:
                                    force_spon = True
                                    break
                cand_spon = get_user_sponsored_offers(uid)
                cand_uns  = user_to_unsponsored_offers.get(uid, pd.DataFrame())
                best_spon = None
                if not cand_spon.empty:
                    for _, row in cand_spon.iterrows():
                        if (uid, row['contract_address']) not in assigned_pairs and remaining.get(row['contract_address'], 0) > 0:
                            best_spon = row
                            break
                best_uns = None
                if not cand_uns.empty:
                    for _, row in cand_uns.iterrows():
                        if (uid, row['contract_address']) not in assigned_pairs and remaining.get(row['contract_address'], 0) > 0:
                            best_uns = row
                            break
                # Choose sponsored when both are viable (small bias toward sponsored)
                if force_spon and best_spon is not None:
                    user_sponsorship[uid] = True
                elif best_spon is not None and best_uns is not None:
                    user_sponsorship[uid] = True
                elif best_spon is not None:
                    user_sponsorship[uid] = True
                elif best_uns is not None:
                    user_sponsorship[uid] = False
                else:
                    # No capacity-matching offers for this user
                    continue
            # Fetch offers in the user's chosen type
            user_offers_df = user_to_sponsored_offers.get(uid, pd.DataFrame()) if user_sponsorship[uid] else user_to_unsponsored_offers.get(uid, pd.DataFrame())
            if user_offers_df is None or user_offers_df.empty:
                continue
            # Try to assign the best next available offer for this user
            assigned_this_user = False
            sponsored_try_limit = 5
            tries = 0
            if user_sponsorship[uid] and force_spon:
                # attempt limited sponsored tries, then fallback to unsponsored within same iteration
                for _, offer in user_offers_df.iterrows():
                    tries += 1
                    if try_assign_offer(offer):
                        made_progress = True
                        assigned_this_user = True
                        break
                    if tries >= sponsored_try_limit:
                        break
                if not assigned_this_user and user_assigned_count.get(uid, 0) < k:
                    # fallback to unsponsored for this iteration
                    cand_uns_df = user_to_unsponsored_offers.get(uid, pd.DataFrame())
                    if cand_uns_df is not None and not cand_uns_df.empty:
                        for _, offer in cand_uns_df.iterrows():
                            if try_assign_offer(offer):
                                user_sponsorship[uid] = False
                                made_progress = True
                                assigned_this_user = True
                                break
            else:
                for _, offer in user_offers_df.iterrows():
                    if try_assign_offer(offer):
                        made_progress = True
                        assigned_this_user = True
                        break
            # If this user had no assignable offers left in their type, skip
        # If a full round made no progress, stop to avoid spinning
        if not made_progress:
            break

    # Dynamic group deficits for prioritization in sweeps
    group_current_now = {}
    group_targets_now = {}
    deficits_by_group = None
    if group_ratios is not None:
        # Targets from ratios
        for _, row in group_ratios.iterrows():
            gid = row['group_id']
            users_in_group = elig[elig['group_id'] == gid]['user_id'].nunique()
            group_targets_now[gid] = int(math.ceil(users_in_group * float(row['sponsorship_ratio'])))
        # Current sponsored users (with >=1 assignment)
        for uid, is_spon in user_sponsorship.items():
            if is_spon and user_assigned_count.get(uid, 0) > 0:
                gid = user_to_group.get(uid)
                if gid is not None:
                    group_current_now[gid] = group_current_now.get(gid, 0) + 1
        deficits_by_group = {gid: max(0, group_targets_now.get(gid, 0) - group_current_now.get(gid, 0)) for gid in group_targets_now.keys()}

    # Post top-up sponsored capacity sweep: try to drain remaining sponsored units
    # Assign to eligible users below k, keeping uniformity and no duplicates
    if remaining:
        # Precompute sponsorship map for contracts
        c_is_spon = caps.set_index('contract_address')['is_sponsored'].to_dict()
        # Iterate sponsored contracts with remaining capacity
        for ca, cap_left in list(remaining.items()):
            if cap_left <= 0:
                continue
            if not c_is_spon.get(ca, False):
                continue  # only sponsored sweep
            # Candidates: sponsored eligibility rows for this contract, sorted by score
            cand_df = sponsored_elig[sponsored_elig['contract_address'] == ca]
            if cand_df.empty:
                continue
            # Prioritize groups with largest sponsorship deficits, then score
            if deficits_by_group is not None:
                cand_df = cand_df.assign(_gid=cand_df['user_id'].map(user_to_group))
                cand_df['_deficit'] = cand_df['_gid'].map(deficits_by_group).fillna(0)
                cand_df = cand_df.sort_values(['_deficit', 'score'], ascending=[False, False])
            for _, offer in cand_df.iterrows():
                if total_deadline and time.time() > total_deadline:
                    break
                if remaining.get(ca, 0) <= 0:
                    break
                uid = offer['user_id']
                # skip users already at k, duplicates, or locked to unsponsored
                if user_assigned_count.get(uid, 0) >= k:
                    continue
                if (uid, ca) in assigned_pairs:
                    continue
                if uid in user_sponsorship and user_sponsorship[uid] is False:
                    continue
                # ensure user sponsorship is sponsored (uniformity)
                if uid not in user_sponsorship:
                    user_sponsorship[uid] = True
                    if deficits_by_group is not None:
                        gid = user_to_group.get(uid)
                        if gid is not None:
                            group_current_now[gid] = group_current_now.get(gid, 0) + 1
                            deficits_by_group[gid] = max(0, group_targets_now.get(gid, 0) - group_current_now.get(gid, 0))
                if try_assign_offer(offer):
                    # already updated counts/capacity
                    continue

    # Conversion sweep: convert fully-unsponsored users to sponsored when enough sponsored capacity exists
    # Keeps all-or-nothing: only flip a user if we can replace ALL of their unsponsored assignments
    if remaining:
        # Users eligible for conversion: currently marked unsponsored and have at least one assignment
        unspon_users = [uid for uid, t in user_sponsorship.items() if t is False and user_assigned_count.get(uid, 0) > 0]
        # Prioritize: fewest assigned first, then biggest group deficits
        if deficits_by_group is not None:
            unspon_users.sort(key=lambda u: (user_assigned_count.get(u, 0), -deficits_by_group.get(user_to_group.get(u), 0)))
        else:
            unspon_users.sort(key=lambda u: user_assigned_count.get(u, 0))
        # Iterate users; respect overall timeout
        for uid in unspon_users:
            if total_deadline and time.time() > total_deadline:
                break
            current_n = user_assigned_count.get(uid, 0)
            if current_n <= 0:
                continue
            # Identify unsponsored assignments to remove for this user
            user_unspon_assigned_contracts = [ca for (u, ca) in assigned_pairs if u == uid and not c_is_spon.get(ca, False)]
            if len(user_unspon_assigned_contracts) != current_n:
                # This user may already be mixed due to prior logic; skip to preserve uniformity
                continue
            # Gather candidate sponsored offers with remaining capacity for this user
            cand_spon_df = user_to_sponsored_offers.get(uid, pd.DataFrame())
            if cand_spon_df is None or cand_spon_df.empty:
                continue
            cand_rows = []
            seen_contracts = set()
            for _, row in cand_spon_df.iterrows():
                ca = row['contract_address']
                if (uid, ca) in assigned_pairs:
                    continue
                if remaining.get(ca, 0) <= 0:
                    continue
                if ca in seen_contracts:
                    continue
                cand_rows.append(row)
                seen_contracts.add(ca)
                if len(cand_rows) >= current_n:
                    break
            # Only proceed if we can replace ALL existing unsponsored picks
            if len(cand_rows) < current_n:
                continue
            # Perform conversion: remove unsponsored assignments, then add sponsored ones
            # Remove unsponsored assignments
            for ca in user_unspon_assigned_contracts:
                if (uid, ca) in assigned_pairs:
                    assigned_pairs.remove((uid, ca))
                    user_assigned_count[uid] = user_assigned_count.get(uid, 1) - 1
                    remaining[ca] = remaining.get(ca, 0) + 1
            # Assign sponsored replacements (count back to previous level)
            for row in cand_rows:
                try_assign_offer(row)
            # If conversion succeeded (no unsponsored left), flip type to sponsored
            # Verify: user has no unsponsored assignments in assigned_pairs
            still_unspon = any((u == uid and not c_is_spon.get(ca, False)) for (u, ca) in assigned_pairs)
            if not still_unspon:
                user_sponsorship[uid] = True
                if deficits_by_group is not None:
                    gid = user_to_group.get(uid)
                    if gid is not None:
                        group_current_now[gid] = group_current_now.get(gid, 0) + 1
                        deficits_by_group[gid] = max(0, group_targets_now.get(gid, 0) - group_current_now.get(gid, 0))
                # Post-conversion top-up: try to fill remaining slots up to k with sponsored offers
                cand_spon_df2 = user_to_sponsored_offers.get(uid, pd.DataFrame())
                if cand_spon_df2 is not None and not cand_spon_df2.empty:
                    for _, row2 in cand_spon_df2.iterrows():
                        if total_deadline and time.time() > total_deadline:
                            break
                        if user_assigned_count.get(uid, 0) >= k:
                            break
                        ca2 = row2['contract_address']
                        if (uid, ca2) in assigned_pairs:
                            continue
                        if remaining.get(ca2, 0) <= 0:
                            continue
                        try_assign_offer(row2)
            else:
                # Rollback not implemented; extremely unlikely due to pre-check len(cand_rows) >= current_n
                # If it happens, leave user type as unsponsored to avoid mixing
                pass

    # Post top-up unsponsored capacity sweep: drain remaining unsponsored units
    # Assign to eligible users below k, keeping uniformity and no duplicates
    if remaining:
        for ca, cap_left in list(remaining.items()):
            if cap_left <= 0:
                continue
            if c_is_spon.get(ca, False):
                continue  # only unsponsored sweep here
            cand_df = unsponsored_elig[unsponsored_elig['contract_address'] == ca]
            if cand_df.empty:
                continue
            for _, offer in cand_df.iterrows():
                if total_deadline and time.time() > total_deadline:
                    break
                if remaining.get(ca, 0) <= 0:
                    break
                uid = offer['user_id']
                # skip users already at k, duplicates, or locked to sponsored
                if user_assigned_count.get(uid, 0) >= k:
                    continue
                if (uid, ca) in assigned_pairs:
                    continue
                if uid in user_sponsorship and user_sponsorship[uid] is True:
                    continue
                # ensure user sponsorship is unsponsored (uniformity)
                if uid not in user_sponsorship:
                    user_sponsorship[uid] = False
                try_assign_offer(offer)
        
    
    # Convert to DataFrame from final assignment pairs to reflect any conversions
    if len(assigned_pairs) == 0:
        result_df = pd.DataFrame(columns=elig.columns)
    else:
        key_series = (elig['user_id'].astype('string') + '|' + elig['contract_address'].astype('string'))
        assigned_keys = set([str(u) + '|' + str(ca) for (u, ca) in assigned_pairs])
        mask = key_series.isin(assigned_keys)
        result_df = elig.loc[mask].copy()
    
    elapsed = time.time() - start_time
    print(f"[greedy] Completed: {len(assigned_pairs):,} assignments, {len(set([u for (u, _) in assigned_pairs])):,} users in {elapsed:.1f}s")
    
    # Print group progress summary
    if group_ratios is not None:
        print(f"[greedy] Group sponsorship progress:")
        # Recompute from final assignments/types
        current_map = {}
        for uid, is_spon in user_sponsorship.items():
            if is_spon and user_assigned_count.get(uid, 0) > 0:
                gid = user_to_group.get(uid)
                if gid is not None:
                    current_map[gid] = current_map.get(gid, 0) + 1
        for _, row in group_ratios.iterrows():
            gid = row['group_id']
            users_in_group = elig[elig['group_id'] == gid]['user_id'].nunique()
            target = int(math.ceil(users_in_group * float(row['sponsorship_ratio'])))
            current = current_map.get(gid, 0)
            actual_ratio = (current / users_in_group) if users_in_group > 0 else 0.0
            print(f"  Group {gid}: {current}/{target} ({actual_ratio:.1%} vs target {row['sponsorship_ratio']:.1%})")
    
    return result_df


def _parse_cov_w(cov_w_str: str, k: int):
    if not cov_w_str: return [0.0] * k
    parts = [float(x.strip()) for x in cov_w_str.split(',') if x.strip()]
    if len(parts) < k: parts += [0.0] * (k - len(parts))
    return parts[:k]


def _hint_indices_from_greedy(elig: pd.DataFrame, g: pd.DataFrame) -> np.ndarray:
    """Vectorized mapping: rows in elig that appear in greedy picks."""
    if g is None or g.empty: return np.empty(0, dtype=np.int64)
    ekeys = (elig['user_id'].astype('string') + '|' + elig['contract_address'].astype('string'))
    gset = set((g['user_id'].astype('string') + '|' + g['contract_address'].astype('string')).tolist())
    mask = ekeys.isin(gset).to_numpy()
    return np.flatnonzero(mask)


# ---- safe sum wrapper for CP-SAT (avoids Python built-in 'sum' ambiguity)
def LSum(items):
    return cp_model.LinearExpr.Sum(list(items))


# ───────────── PuLP (z coverage mode) ─────────────
def ilp_pulp(caps, elig, k, timeout, cov_w_str):
    if not HAVE_PULP: return None, 0.0
    t0 = time.time()

    cov_w = _parse_cov_w(cov_w_str, k)  # e.g., [0.0003,0.0006,0.001]
    prob = pulp.LpProblem("drop", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in elig.index}

    by_user = elig.groupby('user_id').groups
    by_c = elig.groupby('contract_address').groups

    # coverage slot variables z_{u,t}, t = 1..k (1st..kth offer)
    z = {}
    for u in by_user.keys():
        for t in range(1, k+1):
            z[(u, t)] = pulp.LpVariable(f"z_{u}_{t}", cat="Binary")

    # objective = score + sum_u sum_t cov_w[t-1] * z_{u,t} + sponsored bonus
    sponsored_bonus = 0.001  # Small bonus per sponsored assignment (adjust as needed)
    prob += pulp.lpSum(elig.loc[i, 'score'] * x[i] for i in elig.index) + \
            pulp.lpSum(cov_w[t-1] * z[(u, t)] for u in by_user.keys() for t in range(1, k+1)) + \
            pulp.lpSum(sponsored_bonus * x[i] for i in elig.index if elig.loc[i, 'is_sponsored'])

    # per-user ≤ k
    for u, idx in by_user.items():
        prob += pulp.lpSum(x[i] for i in idx) <= k

    # coverage linking: sum_i x_{u,i} >= t * z_{u,t}
    for u, idx in by_user.items():
        su = pulp.lpSum(x[i] for i in idx)
        for t in range(1, k+1):
            prob += su - t * z[(u, t)] >= 0

    # per-contract ≤ cap
    cap = caps.set_index('contract_address').cap_face.to_dict()
    for caddr, idx in by_c.items():
        prob += pulp.lpSum(x[i] for i in idx) <= int(cap[caddr])

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(timeout))
    prob.solve(solver)
    t = time.time() - t0
    status = pulp.LpStatus.get(prob.status, "Unknown")
    obj = float(pulp.value(prob.objective)) if status in ("Optimal", "Feasible") else float('nan')
    print(f"→ PuLP  ILP ……status={status}  time={t:.1f}s  objective={obj:.6f}")
    if status not in ("Optimal", "Feasible"): return None, t

    pick = [i for i in elig.index if x[i].value() > 0.5]
    return elig.loc[pick], t


# ---- AddHint compatibility (handles index property vs Index() method)
def _add_hint_compat(model, x_vars, hint_idx):
    if len(hint_idx) == 0:
        print("[warm-start] no hint rows")
        return
    try:
        # Newer API path (expects var.index property)
        model.AddHint([x_vars[i] for i in hint_idx], [1] * len(hint_idx))
        print(f"[warm-start] hinted {len(hint_idx):,} vars via AddHint()")
        return
    except Exception as e1:
        # Fallback: write into proto using Index() method
        try:
            proto = model._CpModel__model  # CP-SAT python wrapper private field
            proto.solution_hint.vars.extend([x_vars[i].Index() for i in hint_idx])
            proto.solution_hint.values.extend([1] * len(hint_idx))
            print(f"[warm-start] hinted {len(hint_idx):,} vars via proto")
        except Exception as e2:
            print(f"[warm-start] failed to add hints: {e1} / {e2} → continuing without hints")


# ───────────── OR-Tools CP-SAT ─────────────
def ilp_ortools(caps, elig, k, timeout, or_workers, or_log, cov_mode, cov_w_str, shortfall_penalty, warm_start):
    if (not HAVE_OR) or (len(elig) > MAX_CP_SAT_VARS):
        reason = []
        if not HAVE_OR: reason.append("HAVE_OR=False")
        if len(elig) > MAX_CP_SAT_VARS: reason.append(f"len(elig)={len(elig):,} > {MAX_CP_SAT_VARS:,}")
        print("→ Skipping OR-Tools:", "; ".join(reason))
        return None, 0.0

    t0 = time.time()
    m = cp_model.CpModel()

    uid_cat = pd.Categorical(elig.user_id)
    cid_cat = pd.Categorical(elig.contract_address)

    # x vars
    x = [m.NewBoolVar(f"x{i}") for i in range(len(elig))]

    # prebuild index lists
    idx_by_user = {uc: np.where(uid_cat.codes == uc)[0] for uc in range(len(uid_cat.categories))}
    idx_by_c = {cc: np.where(cid_cat.codes == cc)[0] for cc in range(len(cid_cat.categories))}

    # per-user ≤ k
    for uc, idx in idx_by_user.items():
        m.Add(LSum([x[i] for i in idx]) <= k)

    # per-contract ≤ cap
    cap_vec = caps.set_index('contract_address').cap_face.astype(int)
    for cc, idx in idx_by_c.items():
        addr = str(cid_cat.categories[cc])
        m.Add(LSum([x[i] for i in idx]) <= int(cap_vec[addr]))

    # SPONSORSHIP RATIO CONSTRAINTS
    # Create y[u] variables: 1 if user u gets sponsored contracts, 0 if non-sponsored
    y_user = {}
    for uc in range(len(uid_cat.categories)):
        user_id = str(uid_cat.categories[uc])
        y_user[uc] = m.NewBoolVar(f"y_user_{user_id}")
    
    # Sponsorship uniformity constraints: if x[i]=1, then user's sponsorship type must match contract's
    for i in range(len(elig)):
        uc = uid_cat.codes[i]  # user category index
        is_sponsored = elig.iloc[i]['is_sponsored']
        
        if pd.isna(is_sponsored):
            raise ValueError(f"Missing is_sponsored for elig row {i}")
            
        # Convert is_sponsored to boolean
        is_sponsored_bool = is_sponsored in [True, 1, '1', 'true', 'True', 't', 'T']
        
        if is_sponsored_bool:
            # If contract is sponsored, user must be in sponsored group: x[i] ≤ y[uc]
            m.Add(x[i] <= y_user[uc])
        else:
            # If contract is non-sponsored, user must NOT be in sponsored group: x[i] ≤ (1 - y[uc])
            m.Add(x[i] <= (1 - y_user[uc]))
    
    # Group sponsorship ratio constraints
    # Group users by their group_id and enforce ratio constraints
    group_users = elig.groupby('group_id')['user_id'].apply(lambda x: x.unique()).to_dict()
    group_ratios_dict = elig.groupby('group_id')['sponsorship_ratio'].first().to_dict()
    
    for group_id, users_in_group in group_users.items():
        ratio = float(group_ratios_dict[group_id])
        
        # Find user category indices for this group
        group_user_cats = []
        for user_id in users_in_group:
            try:
                uc = uid_cat.categories.get_loc(user_id)
                group_user_cats.append(uc)
            except KeyError:
                # User not in current elig data (filtered out), skip
                continue
        
        if group_user_cats:
            # Sum of sponsored users in group ≥ group_size * ratio (minimum, can exceed)
            group_size = len(group_user_cats)
            min_sponsored = int(math.ceil(group_size * ratio))
            m.Add(LSum([y_user[uc] for uc in group_user_cats]) >= min_sponsored)
            
            print(f"Group {group_id}: {group_size} users, ratio={ratio:.2f}, min_sponsored={min_sponsored}")

    # Link y_user to having at least one assignment
    # A user can only be counted as sponsored if they actually receive at least one assignment
    for uc, idx in idx_by_user.items():
        m.Add(LSum([x[i] for i in idx]) >= y_user[uc])

    # objective terms
    scores = (elig.score * 1_000_000).astype(int).to_numpy()
    obj_terms = [scores[i] * x[i] for i in range(len(elig))]
    
    # Per-user sponsored bonus (not per eligible row)
    # This reflects business preference for sponsored rewards
    sponsored_bonus = 1000  # Small bonus per sponsored user (adjust as needed)
    sponsored_terms = [sponsored_bonus * y_user[uc] for uc in range(len(uid_cat.categories))]

    if cov_mode == "z":
        # coverage z_{u,t}: LSum(x_u) >= t * z_{u,t}
        cov_w = _parse_cov_w(cov_w_str, k)
        cov_terms = []
        for uc, idx in idx_by_user.items():
            su = LSum([x[i] for i in idx])
            for t in range(1, k+1):
                zz = m.NewBoolVar(f"z_{uc}_{t}")
                m.Add(su - t * zz >= 0)
                if cov_w[t-1] != 0.0:
                    cov_terms.append(int(cov_w[t-1] * 1_000_000) * zz)
        m.Maximize(LSum(obj_terms) + LSum(cov_terms) + LSum(sponsored_terms))
    else:
        # shortfall mode: per-user shortfall penalty (lighter/faster)
        pen = int(float(shortfall_penalty) * 1_000_000)  # penalty per missing offer
        short_vars = []
        for uc, idx in idx_by_user.items():
            short_u = m.NewIntVar(0, k, f"short_{uc}")
            m.Add(LSum([x[i] for i in idx]) + short_u >= k)
            short_vars.append(short_u)
        m.Maximize(LSum(obj_terms) - pen * LSum(short_vars) + LSum(sponsored_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout)
    solver.parameters.num_search_workers = max(1, int(or_workers))
    solver.parameters.log_search_progress = bool(or_log)

    # Warm start from greedy (optional)
    if warm_start:
        t_ws = time.time()
        # Get group ratios for greedy warm start
        group_ratios_for_greedy = None
        if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
            group_ratios_for_greedy = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        
        # Extend warm-start time to push ratios further before seeding
        g = greedy(caps, elig, k, seed=42, group_ratios=group_ratios_for_greedy, timeout=150)
        hint_idx = _hint_indices_from_greedy(elig, g)
        _add_hint_compat(m, x, hint_idx)
        print(f"[warm-start] built in {time.time()-t_ws:.1f}s")

    status = solver.Solve(m)
    t = time.time() - t0
    status_str = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}.get(status, str(status))
    obj = solver.ObjectiveValue() / 1_000_000.0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float('nan')
    print(f"→ OR-Tools ILP …status={status_str}  time={t:.1f}s  objective={obj:.6f}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, t

    mask = np.fromiter((solver.Value(xi) for xi in x), bool, len(x))
    return elig[mask], t


def main(cfg):
    t0 = time.time()
    caps, elig, user_groups, group_ratios = load_inputs(
        cfg.caps, cfg.elig, cfg.user_groups, cfg.group_ratios, 
        cfg.unsponsored_cap, cfg.top_n, cfg.min_score
    )
    print(f"caps: {len(caps):,}   elig pairs: {len(elig):,}")
    print(f"groups: {len(group_ratios):,}   users with groups: {len(user_groups):,}")

    df = None; t_used = 0.0
    # Greedy-only mode: skip ILP solvers entirely
    if cfg.solver == "greedy":
        print("→ Greedy only mode")
        group_ratios_for_greedy = None
        if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
            group_ratios_for_greedy = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        df = greedy(caps, elig, cfg.k, cfg.rng, group_ratios=group_ratios_for_greedy, timeout=cfg.timeout)
        label = "Greedy"
        print_summary(df, caps, elig, cfg.k, label, t_used)
        df.to_csv(cfg.out, index=False)
        print(f"✅ wrote {cfg.out}   (total wall time {time.time()-t0:.1f}s)")
        return
    if cfg.solver in ("pulp", "both"):
        df, t_used = ilp_pulp(caps, elig, cfg.k, cfg.timeout, cfg.cov_w)

    if (df is None or df.empty) and (cfg.solver in ("or", "both")):
        df, t_used = ilp_ortools(
            caps, elig, cfg.k, cfg.timeout,
            cfg.or_workers, cfg.or_log,
            cfg.cov_mode, cfg.cov_w, cfg.shortfall_penalty,
            cfg.warm_start
        )

    if df is None or df.empty:
        print("→ Greedy fallback")
        # Get group ratios for greedy fallback
        group_ratios_for_greedy = None
        if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
            group_ratios_for_greedy = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        
        # Use full CLI timeout for greedy fallback
        df = greedy(caps, elig, cfg.k, cfg.rng, group_ratios=group_ratios_for_greedy, timeout=cfg.timeout)
        label = "Greedy"
    else:
        label = "PuLP" if cfg.solver == "pulp" or (cfg.solver == "both" and t_used > 0 and df is not None) else "OR-Tools"

    print_summary(df, caps, elig, cfg.k, label, t_used)
    df.to_csv(cfg.out, index=False)
    print(f"✅ wrote {cfg.out}   (total wall time {time.time()-t0:.1f}s)")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--elig", required=True)
    ap.add_argument("--user_groups", required=True)
    ap.add_argument("--group_ratios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=3, help="offers per user")
    ap.add_argument("--top_n", type=int, default=0, help="pre-trim per user (0=off)")
    ap.add_argument("--unsponsored_cap", type=int, default=10_000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--rng", type=int, default=42)
    ap.add_argument("--solver", choices=("both", "pulp", "or", "greedy"), default="both")
    ap.add_argument("--or_workers", type=int, default=8)
    ap.add_argument("--or_log", action="store_true")
    ap.add_argument("--cov_w", default="0.0003,0.0006,0.001",
                    help="coverage weights for 1st,2nd,3rd offers per user (z mode)")
    ap.add_argument("--cov_mode", choices=("z", "shortfall"), default="z",
                    help="z=classic z_{u,t} coverage; shortfall=penalize missing offers")
    ap.add_argument("--shortfall_penalty", type=float, default=0.1,
                    help="penalty per missing offer (shortfall mode)")
    ap.add_argument("--min_score", type=float, default=None,
                    help="optional filter: drop elig rows with score < min_score")
    ap.add_argument("--warm_start", dest="warm_start", action="store_true", default=True,
                    help="use greedy to AddHint to CP-SAT (default)")
    ap.add_argument("--no_warm_start", dest="warm_start", action="store_false",
                    help="disable warm start")
    cfg = ap.parse_args()
    try:
        main(cfg)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    cli()
