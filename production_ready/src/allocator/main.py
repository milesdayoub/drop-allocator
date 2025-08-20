#!/usr/bin/env python3
"""
Claim – Drop Allocator with coverage encouragement (Production)

What this file contains (no omissions):
• Input loading, validation, and summaries
• Greedy solver (enhanced):
    - Dynamic assignment-level sponsorship ratio guard per group:
        --enforce_assignment_ratio
        --assignment_ratio_guard {none,moving,hard}
        --ratio_slack
    - Third-slot-only mixing, sponsored-first bias
    - Per-group cap on mixed users (bug fix from older version)
    - Per-group cap on "sponsored users" via --max_sponsored_ratio
    - cache_factor for deeper tail search
    - NEW: scarcity weighting nudge for under-utilized sponsored contracts:
        --scarcity_alpha (0..~0.2), effective score = score * (1 + α * remaining/cap)
• ILP solvers preserved:
    - PuLP (CBC) path (coverage z_u,t, caps, ≤k per user)
    - OR-Tools CP-SAT path (caps, ≤k, user-level ratio min, uniformity)
    - Warm start from Greedy (AddHint), identical to older file (now passes scarcity_alpha)
• Same reporting: user-level ratio, assignment-level ratio, capacity analysis

Suggested knobs for your last run:
  --cache_factor 30 \
  --mix_tail \
  --sponsored_first_rounds 2 \
  --third_slot_only_mixing \
  --enforce_assignment_ratio \
  --assignment_ratio_guard hard \
  --ratio_slack 0.0 \
  --max_sponsored_ratio 0.98 \
  --max_mixed_share 0.30 \
  --scarcity_alpha 0.08
"""

from __future__ import annotations
import argparse, sys, time, textwrap, math, logging
import pandas as pd, numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

    # CRITICAL: Filter to only users with groups
    logger.info(f"Before user filtering: {len(elig):,} elig pairs, {elig['user_id'].nunique():,} unique users")
    elig = elig[elig.user_id.isin(user_groups.user_id)]
    logger.info(f"After user filtering: {len(elig):,} elig pairs, {elig['user_id'].nunique():,} unique users")
    
    # Keep only contracts with positive cap after synthetic fill
    elig = elig[elig.contract_address.isin(caps.contract_address)]

    # Optional: drop very low/zero scores to shrink model
    if min_score is not None:
        elig = elig[elig['score'] >= float(min_score)]

    # Optional: pre-trim per user to top-N contracts by score
    if top_n is not None and int(top_n) > 0:
        elig = (
            elig.sort_values(['user_id', 'score'], ascending=[True, False])
                .groupby('user_id', group_keys=False)
                .head(int(top_n))
                .reset_index(drop=True)
        )

    # Add contract sponsorship information
    caps_spon = caps[['contract_address', 'is_sponsored']].copy()
    elig = elig.merge(caps_spon, on='contract_address', how='left')
    
    # Add group info
    elig = elig.merge(user_groups[['user_id', 'group_id']], on='user_id', how='left')
    
    # Add sponsorship ratio by group
    elig = elig.merge(group_ratios[['group_id', 'sponsorship_ratio']], on='group_id', how='left')
    
    # Validate
    if elig['group_id'].isna().any():
        missing = elig[elig['group_id'].isna()]
        raise ValueError(f"Found {len(missing)} elig pairs with missing group_id")
    if elig['sponsorship_ratio'].isna().any():
        missing = elig[elig['sponsorship_ratio'].isna()]
        raise ValueError(f"Found {len(missing)} elig pairs with missing sponsorship_ratio")
    
    return caps.reset_index(drop=True), elig.reset_index(drop=True), user_groups.reset_index(drop=True), group_ratios.reset_index(drop=True)


def summarize(df_assign: pd.DataFrame, elig: pd.DataFrame, k: int):
    users = pd.Index(elig['user_id'].unique())
    per_user = df_assign.groupby('user_id').size().reindex(users, fill_value=0)
    dist = per_user.value_counts().reindex(range(0, k+1), fill_value=0).to_dict()
    fill = per_user.sum() / (len(users) * k) if len(users) and k else 0.0
    return {'n_users': int(len(users)),
            'dist': {int(i): int(dist[i]) for i in dist},
            'fill_rate': float(fill)}


def analyze_sponsored_capacity(df_assign, caps):
    """Analyze sponsored contract capacity utilization"""
    logger.info("=== SPONSORSHIP CAPACITY ANALYSIS ===")
    
    # Get contract caps for sponsored contracts
    sponsored_caps = caps[caps['is_sponsored'] == True]
    total_sponsored_cap = int(sponsored_caps['cap_face'].sum())
    
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
    
    logger.info(f"Total sponsored contract capacity: {total_sponsored_cap:,}")
    logger.info(f"Total sponsored assignments made: {len(sponsored_assignments):,}")
    if total_sponsored_cap > 0:
        logger.info(f"Overall sponsored capacity utilization: {len(sponsored_assignments)/total_sponsored_cap*100:.1f}%")
    
    # Show contracts that are fully utilized
    fully_utilized = capacity_analysis[capacity_analysis['assigned'] >= capacity_analysis['cap_face']]
    if not fully_utilized.empty:
        logger.warning(f"🚨 FULLY UTILIZED sponsored contracts ({len(fully_utilized)}):")
        logger.info(fully_utilized[['contract_address', 'cap_face', 'assigned', 'utilization_pct']].head(10).to_string())
    
    # Show contracts with remaining capacity
    remaining_capacity = capacity_analysis[capacity_analysis['remaining'] > 0]
    if not remaining_capacity.empty:
        logger.info(f"✅ Sponsored contracts with remaining capacity ({len(remaining_capacity)}):")
        logger.info(remaining_capacity[['contract_address', 'cap_face', 'assigned', 'remaining', 'utilization_pct']].head(10).to_string())
    
    # Show contracts with no assignments
    unused = capacity_analysis[capacity_analysis['assigned'] == 0]
    if not unused.empty:
        logger.warning(f"⚠️  Unused sponsored contracts ({len(unused)}):")
        logger.info(unused[['contract_address', 'cap_face']].head(10).to_string())
    
    # Summary
    logger.info("=== CAPACITY SUMMARY ===")
    logger.info(f"Fully utilized: {len(fully_utilized)} contracts")
    logger.info(f"Partially used: {len(remaining_capacity)} contracts") 
    logger.info(f"Unused: {len(unused)} contracts")
    logger.info(f"Total sponsored contracts: {len(sponsored_caps)}")


def validate_sponsorship_ratios(df_assign, elig, group_ratios, caps=None):
    if df_assign.empty:
        return
    
    logger.info("=== SPONSORSHIP RATIO VALIDATION ===")
    
    # Merge to get group and sponsorship info
    df = df_assign.merge(
        elig[['user_id', 'contract_address', 'group_id', 'sponsorship_ratio', 'is_sponsored']], 
        on=['user_id', 'contract_address'], 
        how='left',
        suffixes=('', '_elig'),
    )

    # 1) Per-user uniformity (diagnostic; may be relaxed by policy)
    per_user = df.groupby('user_id')['is_sponsored'].agg(['nunique', 'first'])
    mixed_users = per_user[per_user['nunique'] > 1]
    if not mixed_users.empty:
        logger.error(f"❌ VIOLATION: {len(mixed_users)} users have mixed sponsorship types!")
        logger.info(mixed_users.head().to_string())
    else:
        logger.info("✅ All users have uniform sponsorship types")
    
    # 2) Group-level ratios at USER level (not assignment rows)
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
        logger.error(f"❌ USER-RATIO VIOLATIONS in {len(violations)} groups:")
        logger.info(violations[['total_users', 'sponsored_users', 'expected_min_sponsored', 'sponsorship_ratio', 'actual_ratio']].to_string())
    else:
        logger.info("✅ All groups respect sponsorship ratio constraints")
    
    # 3) Assignment-level ratios (sponsored assignments / total assignments) per group
    df['_spon'] = df['is_sponsored'].astype(bool)
    assign_summary = (
        df.groupby('group_id')
          .agg(total_assignments=('user_id', 'size'),
               sponsored_assignments=('_spon', 'sum'))
          .reset_index()
    )
    assign_summary = assign_summary.merge(ratios, on='group_id', how='left')
    assign_summary['assignment_ratio'] = assign_summary['sponsored_assignments'] / assign_summary['total_assignments'].clip(lower=1)

    logger.info("Assignments-Level Summary:")
    logger.info(assign_summary[['total_assignments', 'sponsored_assignments', 'sponsorship_ratio', 'assignment_ratio']].round(3).to_string())

    viol = assign_summary[assign_summary['assignment_ratio'] + 1e-12 < assign_summary['sponsorship_ratio']]
    if not viol.empty:
        logger.error(f"❌ ASSIGNMENT-RATIO VIOLATIONS in {len(viol):,} groups.")
    else:
        logger.info("✅ All groups respect assignments-level sponsorship ratios")
    
    # Capacity analysis if caps provided
    if caps is not None:
        analyze_sponsored_capacity(df_assign, caps)


def print_summary(df_assign, caps, elig, k, label, t_sec):
    used = df_assign.contract_address.value_counts() if not df_assign.empty else pd.Series(dtype=int)
    cap = caps.set_index('contract_address').cap_face.reindex(used.index).fillna(0)
    cap_viol = used[used > cap]
    stats = summarize(df_assign, elig, k)
    obj = df_assign['score'].sum() if 'score' in df_assign.columns and not df_assign.empty else float('nan')
    
    summary_text = textwrap.dedent(f"""
        ── {label} summary ───────────────────────────────────────
        users (elig)      : {stats['n_users']:,}
        assignments       : {len(df_assign):,}
        Σ score           : {obj:.6f}
        offer counts      : 3→{stats['dist'].get(3,0):,}   2→{stats['dist'].get(2,0):,}   1→{stats['dist'].get(1,0):,}   0→{stats['dist'].get(0,0):,}
        fill rate         : {stats['fill_rate']*100:.2f} %  (of {k} offers per user)
        cap violations    : {'0' if cap_viol.empty else cap_viol.to_dict()}
        wall time         : {t_sec:.1f}s
        ──────────────────────────────────────────────────────────
    """).strip()
    
    logger.info(summary_text)
    
    if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
        group_ratios = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        validate_sponsorship_ratios(df_assign, elig, group_ratios, caps)


# =====================================================================
# Greedy (enhanced): dynamic ratio guard + per-group mixed cap + 3rd-slot mixing
#                    + scarcity weighting for sponsored contracts
# =====================================================================
def greedy(
    caps, elig, k, seed=42,
    group_ratios=None, timeout=None,
    cache_factor=10, mix_tail=False, sponsored_first_rounds=2,
    third_slot_only_mixing=True,
    enforce_assignment_ratio=False, assignment_ratio_guard="hard",
    ratio_slack=0.0, unspon_overflow_pct=0.0,
    max_sponsored_ratio=None, max_mixed_share=1.0,
    scarcity_alpha: float = 0.0
):
    start_time = time.time()
    rng = np.random.default_rng(seed)
    logger.info(f"[greedy] Pre-processing data for {len(elig):,} elig pairs...")

    # Remaining capacity per contract
    remaining = caps.set_index('contract_address').cap_face.to_dict()
    c_is_spon = caps.set_index('contract_address')['is_sponsored'].to_dict()
    cap_face_map = caps.set_index('contract_address').cap_face.to_dict()

    # Pre-sort eligibility by score (desc)
    sponsored_elig = elig[elig['is_sponsored'] == True].copy().sort_values('score', ascending=False)
    unsponsored_elig = elig[elig['is_sponsored'] == False].copy().sort_values('score', ascending=False)

    # Fast per-user caches
    user_to_group = elig.groupby('user_id')['group_id'].first().to_dict()
    user_to_sponsored_offers = {uid: grp.nlargest(k * cache_factor, 'score') for uid, grp in sponsored_elig.groupby('user_id')}
    user_to_unsponsored_offers = {uid: grp.nlargest(k * cache_factor, 'score') for uid, grp in unsponsored_elig.groupby('user_id')}

    # Group metadata
    all_gids = list(elig['group_id'].unique())
    gid_users = {gid: set(elig[elig['group_id']==gid]['user_id'].unique().tolist()) for gid in all_gids}
    gid_ratio  = {row['group_id']: float(row['sponsorship_ratio']) for _, row in group_ratios.iterrows()} if group_ratios is not None else {}
    gid_usercnt = {gid: len(users) for gid, users in gid_users.items()}

    # Trackers
    assigned_pairs = set()      # (user_id, contract_address)
    user_assigned_count = {}    # user -> total assigned
    user_sponsored_count = {}   # user -> sponsored assigned
    user_sponsorship = {}       # user -> True(spon) / False(unspon)
    mixed_users_by_gid = {gid: set() for gid in all_gids}

    # Group-level counters (assignment level)
    gid_assign_total = {gid: 0 for gid in all_gids}
    gid_assign_unspon = {gid: 0 for gid in all_gids}
    gid_sponsored_users = {gid: 0 for gid in all_gids}      # user-level count (≥1 spon)
    gid_sponsored_user_set = {gid: set() for gid in all_gids}

    # Optional cap on number of sponsored users (user-level)
    gid_sponsored_user_cap = None
    if max_sponsored_ratio is not None:
        gid_sponsored_user_cap = {gid: int(math.floor(gid_usercnt[gid] * float(max_sponsored_ratio))) for gid in all_gids}

    # --- scarcity weighting helpers ---
    def _eff_score(row, scarcity_alpha: float, remaining: dict, cap_face_map: dict) -> float:
        """Effective score for sponsored offers with remaining headroom."""
        try:
            if scarcity_alpha <= 0.0 or not bool(row['is_sponsored']):
                return float(row['score'])
            ca = row['contract_address']
            cap = float(cap_face_map.get(ca, 0) or 0)
            if cap <= 0:
                return float(row['score'])
            pressure = max(0.0, min(1.0, (remaining.get(ca, 0) or 0) / cap))
            return float(row['score']) * (1.0 + scarcity_alpha * pressure)
        except Exception:
            # fail-open to original score
            return float(row['score'])

    # Helper: checks if an unsponsored pick is allowed under the chosen guard
    def can_place_unsponsored_in_group(gid: str) -> bool:
        if not enforce_assignment_ratio:
            return True
        r = gid_ratio.get(gid, 0.0)
        T = gid_assign_total[gid]
        U = gid_assign_unspon[gid]
        S = T - U  # sponsored assignments so far

        guard = (assignment_ratio_guard or "none").lower()
        if guard == "none":
            return True

        if guard == "hard":
            # After placing one unsponsored: S / (T+1) >= r - ratio_slack
            # i.e., S >= (r - slack) * (T+1)
            return S >= (r - float(ratio_slack)) * (T + 1 + 1e-9)

        # moving: U <= ((1-r)/r) * S  + slack_abs*(T)
        if r <= 1e-9:  # degenerate (all unsponsored allowed)
            return True
        max_unspon_now = ((1.0 - r) / r) * S + float(ratio_slack) * max(1.0, T)
        return (U + 1) <= max_unspon_now

    def would_exceed_sponsored_user_cap(gid: str, uid: str) -> bool:
        if gid_sponsored_user_cap is None:
            return False
        if uid in gid_sponsored_user_set[gid]:
            return False
        return gid_sponsored_users[gid] >= gid_sponsored_user_cap[gid]

    def bump_group_counters(uid: str, gid: str, is_spon: bool):
        gid_assign_total[gid] += 1
        if not is_spon:
            gid_assign_unspon[gid] += 1
        if is_spon and uid not in gid_sponsored_user_set[gid]:
            gid_sponsored_user_set[gid].add(uid)
            gid_sponsored_users[gid] += 1

    def try_assign_offer(offer_row) -> bool:
        """Central gate that enforces: caps, duplicates, per-user k, assignment ratio guard, third-slot-only mixing, per-group mixed cap, sponsored-user cap."""
        uid = offer_row['user_id']; ca = offer_row['contract_address']
        gid = offer_row['group_id']; is_spon_offer = bool(offer_row['is_sponsored'])

        # Cap & duplicates
        if remaining.get(ca, 0) <= 0: return False
        if (uid, ca) in assigned_pairs: return False
        if user_assigned_count.get(uid, 0) >= k: return False

        # Assignment-level ratio guard for UNSPON
        if (not is_spon_offer) and (not can_place_unsponsored_in_group(gid)):
            return False

        # Third-slot-only mixing rules
        utype = user_sponsorship.get(uid, None)
        u_assigned = user_assigned_count.get(uid, 0)
        u_spon = user_sponsored_count.get(uid, 0)

        # If user type is unknown and offer is sponsored, check sponsored-user cap (if any)
        if utype is None and is_spon_offer and would_exceed_sponsored_user_cap(gid, uid):
            return False

        # Mixing control
        if utype is None:
            pass  # establishing type via this assignment is allowed
        else:
            if utype and (not is_spon_offer):
                # user is 'sponsored-type' but offer is unsponsored
                if not mix_tail:
                    return False
                if third_slot_only_mixing:
                    # Only allow for last slot AND after they have the required sponsored-first rounds
                    if not (u_assigned == k - 1 and u_spon >= min(sponsored_first_rounds, k - 1)):
                        return False
            if (not utype) and is_spon_offer:
                # user is 'unsponsored-type' but offer is sponsored (avoid mixing inward)
                if u_assigned > 0:
                    return False  # keep them pure unless establishing type

        # Mixed users per-group cap
        will_be_mixed = False
        if utype is not None and (utype != is_spon_offer):
            will_be_mixed = True
        if will_be_mixed:
            group_size = max(1, len(gid_users.get(gid, [])))
            if (len(mixed_users_by_gid[gid]) + (0 if uid in mixed_users_by_gid[gid] else 1)) / group_size > float(max_mixed_share):
                return False

        # All guards passed → assign
        assigned_pairs.add((uid, ca))
        remaining[ca] -= 1
        user_assigned_count[uid] = u_assigned + 1
        if is_spon_offer:
            user_sponsored_count[uid] = u_spon + 1

        # Establish user type if unknown
        if utype is None:
            user_sponsorship[uid] = is_spon_offer
            # bump user-level sponsored counts if we just made them sponsored
            if is_spon_offer:
                if uid not in gid_sponsored_user_set[gid]:
                    gid_sponsored_user_set[gid].add(uid)
                    gid_sponsored_users[gid] += 1
        else:
            if will_be_mixed:
                mixed_users_by_gid[gid].add(uid)

        # assignment-level counters
        bump_group_counters(uid, gid, is_spon_offer)
        return True

    # Utility: iterate a user's best available offers for a given type (with optional scarcity weighting for sponsored)
    def iter_user_offers(uid: str, sponsored: bool, scarcity_alpha_local: float = 0.0):
        df = user_to_sponsored_offers.get(uid) if sponsored else user_to_unsponsored_offers.get(uid)
        if df is None or df.empty:
            return
        if sponsored and scarcity_alpha_local > 0.0:
            tmp = df.copy()
            tmp['__eff'] = tmp.apply(lambda r: _eff_score(r, scarcity_alpha_local, remaining, cap_face_map), axis=1)
            tmp = tmp.sort_values('__eff', ascending=False)
            for _, r in tmp.iterrows():
                yield r
        else:
            for _, r in df.iterrows():
                yield r

    # ── Phase 1: meet user-level sponsorship minima (make enough users 'sponsored-type')
    if group_ratios is not None:
        logger.info(f"[greedy] Phase 1: Meeting sponsorship minima for {len(group_ratios)} groups")
        group_targets = {row['group_id']: int(math.ceil(gid_usercnt[row['group_id']] * float(row['sponsorship_ratio'])))
                         for _, row in group_ratios.iterrows()}
        for gid, target in sorted(group_targets.items(), key=lambda kv: kv[1], reverse=True):
            if gid_sponsored_users[gid] >= target: continue
            users = list(gid_users[gid])
            scored = []
            for uid in users:
                df = user_to_sponsored_offers.get(uid)
                if df is None or df.empty: continue
                best = df['score'].max()
                scored.append((uid, best))
            scored.sort(key=lambda x: x[1], reverse=True)
            for uid, _ in scored:
                if gid_sponsored_users[gid] >= target: break
                df = user_to_sponsored_offers.get(uid)
                if df is None or df.empty: continue
                for _, offer in df.iterrows():
                    if offer['group_id'] != gid or not offer['is_sponsored']: continue
                    if try_assign_offer(offer):
                        break

    # ── Phase 2: round-robin top-up to k offers (respect type; allow only 3rd-slot mixing)
    logger.info(f"[greedy] Phase 2: Round-robin top-up to {k} offers")
    all_users = list(elig['user_id'].unique())
    # Shuffle user order to avoid systematic bias and use provided seed
    if len(all_users) > 1:
        all_users = list(rng.permutation(all_users))
    for round_idx in range(1, k+1):
        made_progress = False
        for uid in all_users:
            if user_assigned_count.get(uid, 0) >= round_idx: continue
            gid = user_to_group.get(uid)

            utype = user_sponsorship.get(uid, None)
            # Decide type if unknown: prefer sponsored unless cap blocks
            if utype is None:
                if gid is not None and would_exceed_sponsored_user_cap(gid, uid):
                    user_sponsorship[uid] = False
                    utype = False
                else:
                    user_sponsorship[uid] = True
                    utype = True

            if utype:  # sponsored-type
                placed = False
                for r in iter_user_offers(uid, True, scarcity_alpha):
                    if try_assign_offer(r):
                        placed = True; break
                if not placed and mix_tail:
                    if third_slot_only_mixing and round_idx == k and user_sponsored_count.get(uid, 0) >= min(sponsored_first_rounds, k-1):
                        for r in iter_user_offers(uid, False, scarcity_alpha):
                            if try_assign_offer(r):
                                placed = True; break
                made_progress |= placed
            else:  # unsponsored-type
                placed = False
                for r in iter_user_offers(uid, False, scarcity_alpha):
                    if try_assign_offer(r):
                        placed = True; break
                made_progress |= placed

        if not made_progress:
            break

    # ── Sponsored sweep: drain remaining sponsored capacity toward under-filled users
            logger.info("[greedy] Sponsored sweep (pressure-first)")
    spon_contracts = [ca for ca in remaining if c_is_spon.get(ca, False) and remaining[ca] > 0]
    spon_contracts.sort(key=lambda ca: remaining[ca], reverse=True)
    for ca in spon_contracts:
        cand = sponsored_elig[sponsored_elig['contract_address'] == ca]
        if cand.empty: continue
        for _, offer in cand.iterrows():
            uid = offer['user_id']
            if user_assigned_count.get(uid, 0) >= k: continue
            if user_sponsorship.get(uid, None) is False:
                continue
            try_assign_offer(offer)
            if remaining[ca] <= 0: break

    # ── Conversion sweep: turn fully-unsponsored users into sponsored when possible
            logger.info("[greedy] Conversion sweep (unsponsored → sponsored where possible)")
    unspon_users = [u for u, t in user_sponsorship.items() if t is False and user_assigned_count.get(u,0) > 0]
    unspon_users.sort(key=lambda u: user_assigned_count.get(u,0))
    for uid in unspon_users:
        gid = user_to_group.get(uid)
        if would_exceed_sponsored_user_cap(gid, uid): continue
        need = user_assigned_count.get(uid, 0)
        picks = []
        df = user_to_sponsored_offers.get(uid)
        if df is None or df.empty: continue
        seen = set()
        for _, r in df.iterrows():
            ca = r['contract_address']
            if remaining.get(ca,0) <= 0: continue
            if (uid, ca) in assigned_pairs: continue
            if ca in seen: continue
            picks.append(r); seen.add(ca)
            if len(picks) >= need: break
        if len(picks) < need: continue
        # remove all current (unsponsored) assignments for this user
        to_remove = [ca for (u,ca) in list(assigned_pairs) if u == uid and not c_is_spon.get(ca, False)]
        for ca in to_remove:
            assigned_pairs.remove((uid, ca))
            remaining[ca] = remaining.get(ca,0) + 1
            user_assigned_count[uid] -= 1
            gid_assign_total[gid] -= 1
            gid_assign_unspon[gid] -= 1
        # assign sponsored replacements
        ok = True
        for r in picks:
            if not try_assign_offer(r):
                ok = False; break
        if ok:
            user_sponsorship[uid] = True  # converted

    # ── Unsponsored sweep: place unsponsored for users below k (respect guards)
            logger.info("[greedy] Unsponsored sweep (pressure-first)")
    unspon_contracts = [ca for ca in remaining if not c_is_spon.get(ca, False) and remaining[ca] > 0]
    unspon_contracts.sort(key=lambda ca: remaining[ca], reverse=True)
    for ca in unspon_contracts:
        cand = unsponsored_elig[unsponsored_elig['contract_address'] == ca]
        if cand.empty: continue
        for _, offer in cand.iterrows():
            uid = offer['user_id']; gid = offer['group_id']
            if user_assigned_count.get(uid,0) >= k: continue
            if user_sponsorship.get(uid, None) is True:
                continue  # keep purity; third-slot handled later
            try_assign_offer(offer)
            if remaining[ca] <= 0: break

    # ── Coverage sweep: fill to k; prefer sponsored; third-slot unspon only if allowed & guard passes
            logger.info("[greedy] Coverage sweep (fill to k; mixed types allowed)")
    all_users = list(elig['user_id'].unique())
    underfilled = [u for u in all_users if user_assigned_count.get(u,0) < k]
    for uid in underfilled:
        need = k - user_assigned_count.get(uid,0)
        if need <= 0: continue
        for _ in range(need):
            placed = False
            # try sponsored first (scarcity-weighted)
            for r in iter_user_offers(uid, True, scarcity_alpha):
                if try_assign_offer(r):
                    placed = True; break
            if not placed and mix_tail:
                # Allow unsponsored only for final slot condition (third-slot-only)
                if third_slot_only_mixing and user_assigned_count.get(uid,0) == k - 1 and user_sponsored_count.get(uid,0) >= min(sponsored_first_rounds, k-1):
                    for r in iter_user_offers(uid, False, scarcity_alpha):
                        if try_assign_offer(r):
                            placed = True; break
            if not placed:
                break

    # Build final DataFrame from assigned_pairs
    if len(assigned_pairs) == 0:
        result_df = pd.DataFrame(columns=elig.columns)
    else:
        keys = set([f"{u}|{ca}" for (u,ca) in assigned_pairs])
        ekeys = (elig['user_id'].astype('string') + '|' + elig['contract_address'].astype('string'))
        mask = ekeys.isin(keys)
        result_df = elig.loc[mask].copy()

    elapsed = time.time() - start_time
    logger.info(f"[greedy] Completed: {len(assigned_pairs):,} assignments, {len(set([u for (u, _) in assigned_pairs])):,} users in {elapsed:.1f}s")
    return result_df


# =====================================================================
# ILP helpers and solvers (preserved)
# =====================================================================
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


def LSum(items):
    return cp_model.LinearExpr.Sum(list(items))


def ilp_pulp(caps, elig, k, timeout, cov_w_str):
    if not HAVE_PULP: return None, 0.0
    t0 = time.time()

    cov_w = _parse_cov_w(cov_w_str, k)  # e.g., [0.0003,0.0006,0.001]
    prob = pulp.LpProblem("claim", pulp.LpMaximize)

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
    logger.info(f"→ PuLP  ILP ……status={status}  time={t:.1f}s  objective={obj:.6f}")
    if status not in ("Optimal", "Feasible"): return None, t

    pick = [i for i in elig.index if x[i].value() > 0.5]
    return elig.loc[pick], t


def _add_hint_compat(model, x_vars, hint_idx):
    if len(hint_idx) == 0:
        logger.info("[warm-start] no hint rows"); return
    try:
        model.AddHint([x_vars[i] for i in hint_idx], [1] * len(hint_idx))
        logger.info(f"[warm-start] hinted {len(hint_idx):,} vars via AddHint()")
        return
    except Exception as e1:
        try:
            proto = model._CpModel__model
            proto.solution_hint.vars.extend([x_vars[i].Index() for i in hint_idx])
            proto.solution_hint.values.extend([1] * len(hint_idx))
            logger.info(f"[warm-start] hinted {len(hint_idx):,} vars via proto")
        except Exception as e2:
            logger.warning(f"[warm-start] failed to add hints: {e1} / {e2} → continuing without hints")


def ilp_ortools(caps, elig, k, timeout, or_workers, or_log, cov_mode, cov_w_str, shortfall_penalty, warm_start, scarcity_alpha_ws: float = 0.0):
    if (not HAVE_OR) or (len(elig) > MAX_CP_SAT_VARS):
        reason = []
        if not HAVE_OR: reason.append("HAVE_OR=False")
        if len(elig) > MAX_CP_SAT_VARS: reason.append(f"len(elig)={len(elig):,} > {MAX_CP_SAT_VARS:,}")
        logger.warning("→ Skipping OR-Tools: %s", "; ".join(reason))
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

    # SPONSORSHIP RATIO CONSTRAINTS (user-level minima) + uniformity
    y_user = {}
    for uc in range(len(uid_cat.categories)):
        user_id = str(uid_cat.categories[uc])
        y_user[uc] = m.NewBoolVar(f"y_user_{user_id}")
    
    for i in range(len(elig)):
        uc = uid_cat.codes[i]
        is_sponsored = elig.iloc[i]['is_sponsored']
        if pd.isna(is_sponsored):
            raise ValueError(f"Missing is_sponsored for elig row {i}")
        is_sponsored_bool = is_sponsored in [True, 1, '1', 'true', 'True', 't', 'T']
        if is_sponsored_bool:
            m.Add(x[i] <= y_user[uc])
        else:
            m.Add(x[i] <= (1 - y_user[uc]))
    
    group_users = elig.groupby('group_id')['user_id'].apply(lambda x: x.unique()).to_dict()
    group_ratios_dict = elig.groupby('group_id')['sponsorship_ratio'].first().to_dict()
    
    for group_id, users_in_group in group_users.items():
        ratio = float(group_ratios_dict[group_id])
        group_user_cats = []
        for user_id in users_in_group:
            try:
                uc = uid_cat.categories.get_loc(user_id)
                group_user_cats.append(uc)
            except KeyError:
                continue
        if group_user_cats:
            group_size = len(group_user_cats)
            min_sponsored = int(math.ceil(group_size * ratio))
            m.Add(LSum([y_user[uc] for uc in group_user_cats]) >= min_sponsored)
            logger.info(f"Group {group_id}: {group_size} users, ratio={ratio:.2f}, min_sponsored={min_sponsored}")

    for uc, idx in idx_by_user.items():
        m.Add(LSum([x[i] for i in idx]) >= y_user[uc])

    # objective terms
    scores = (elig.score * 1_000_000).astype(int).to_numpy()
    obj_terms = [scores[i] * x[i] for i in range(len(elig))]
    
    sponsored_bonus = 1000  # Small bonus per sponsored user (adjust as needed)
    sponsored_terms = [sponsored_bonus * y_user[uc] for uc in range(len(uid_cat.categories))]

    if cov_mode == "z":
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
        group_ratios_for_greedy = None
        if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
            group_ratios_for_greedy = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        g = greedy(caps, elig, k, seed=42, group_ratios=group_ratios_for_greedy, timeout=150, scarcity_alpha=scarcity_alpha_ws)
        hint_idx = _hint_indices_from_greedy(elig, g)
        _add_hint_compat(m, x, hint_idx)
        logger.info(f"[warm-start] built in {time.time()-t_ws:.1f}s")

    status = solver.Solve(m)
    t = time.time() - t0
    status_str = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}.get(status, str(status))
    obj = solver.ObjectiveValue() / 1_000_000.0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float('nan')
    logger.info(f"→ OR-Tools ILP …status={status_str}  time={t:.1f}s  objective={obj:.6f}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, t

    mask = np.fromiter((solver.Value(xi) for xi in x), bool, len(x))
    return elig[mask], t


# =====================================================================
# Driver
# =====================================================================
def main(cfg):
    t0 = time.time()
    caps, elig, user_groups, group_ratios = load_inputs(
        cfg.caps, cfg.elig, cfg.user_groups, cfg.group_ratios, 
        cfg.unsponsored_cap, cfg.top_n, cfg.min_score
    )
    logger.info(f"caps: {len(caps):,}   elig pairs: {len(elig):,}")
    logger.info(f"groups: {len(group_ratios):,}   users with groups: {len(user_groups):,}")

    df = None; t_used = 0.0
    label = "Greedy"

    if cfg.solver in ("pulp", "both"):
        df, t_used = ilp_pulp(caps, elig, cfg.k, cfg.timeout, cfg.cov_w)
        label = "PuLP"

    if (df is None or df.empty) and (cfg.solver in ("or", "both")):
        df, t_used = ilp_ortools(
            caps, elig, cfg.k, cfg.timeout,
            cfg.or_workers, cfg.or_log,
            cfg.cov_mode, cfg.cov_w, cfg.shortfall_penalty,
            cfg.warm_start,
            cfg.scarcity_alpha  # pass α to warm-start greedy
        )
        label = "OR-Tools"

    if df is None or df.empty or (cfg.solver == "greedy"):
        logger.info("→ Greedy mode")
        group_ratios_for_greedy = None
        if 'group_id' in elig.columns and 'sponsorship_ratio' in elig.columns:
            group_ratios_for_greedy = elig.groupby('group_id')['sponsorship_ratio'].first().reset_index()
        df = greedy(
            caps, elig, cfg.k, cfg.rng,
            group_ratios=group_ratios_for_greedy, timeout=cfg.timeout,
            cache_factor=cfg.cache_factor,
            mix_tail=cfg.mix_tail,
            sponsored_first_rounds=cfg.sponsored_first_rounds,
            third_slot_only_mixing=cfg.third_slot_only_mixing,
            enforce_assignment_ratio=cfg.enforce_assignment_ratio,
            assignment_ratio_guard=cfg.assignment_ratio_guard,
            ratio_slack=cfg.ratio_slack,
            unspon_overflow_pct=cfg.unspon_overflow_pct,
            max_sponsored_ratio=cfg.max_sponsored_ratio,
            max_mixed_share=cfg.max_mixed_share,
            scarcity_alpha=cfg.scarcity_alpha
        )
        label = "Greedy"

    print_summary(df, caps, elig, cfg.k, label, t_used)
    df.to_csv(cfg.out, index=False)
    logger.info(f"✅ wrote {cfg.out}   (total wall time {time.time()-t0:.1f}s)")


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
    ap.add_argument("--solver", choices=("both","pulp","or","greedy"), default="both")

    # OR-Tools / PuLP coverage knobs (preserved)
    ap.add_argument("--or_workers", type=int, default=8)
    ap.add_argument("--or_log", action="store_true")
    ap.add_argument("--cov_w", default="0.0003,0.0006,0.001", help="coverage weights for 1st,2nd,3rd offers per user (z mode)")
    ap.add_argument("--cov_mode", choices=("z", "shortfall"), default="z", help="z=classic z_{u,t} coverage; shortfall=penalize missing offers")
    ap.add_argument("--shortfall_penalty", type=float, default=0.1, help="penalty per missing offer (shortfall mode)")
    ap.add_argument("--warm_start", dest="warm_start", action="store_true", default=True, help="use greedy to AddHint to CP-SAT (default)")
    ap.add_argument("--no_warm_start", dest="warm_start", action="store_false", help="disable warm start")

    # Greedy coverage / policy knobs (new + preserved)
    ap.add_argument("--cache_factor", type=int, default=10, help="per-user cached candidates per type = k * cache_factor")
    ap.add_argument("--mix_tail", action="store_true", help="allow mixing types for coverage (third slot only by default)")
    ap.add_argument("--sponsored_first_rounds", type=int, default=2, help="number of sponsored slots required before unsponsored fallback is allowed")
    ap.add_argument("--third_slot_only_mixing", action="store_true", default=True, help="restrict mixing to 3rd slot only")
    ap.add_argument("--max_sponsored_ratio", type=float, default=None, help="max share of users per group that may be counted as sponsored (user-level)")
    ap.add_argument("--max_mixed_share", type=float, default=1.0, help="max share of users that may be mixed within each group (0..1)")

    # Assignment-level ratio guard (new)
    ap.add_argument("--enforce_assignment_ratio", action="store_true", help="enforce assignment-level sponsorship ratios per group")
    ap.add_argument("--assignment_ratio_guard", choices=("none","moving","hard"), default="hard",
                    help="unsponsored guard type: hard=no below-target placements; moving=proportional budget; none=off")
    ap.add_argument("--ratio_slack", type=float, default=0.0, help="absolute slack for ratio guard (e.g., 0.002 allows -0.2pp)")
    ap.add_argument("--unspon_overflow_pct", type=float, default=0.0, help="(legacy) extra unsponsored fraction when using moving guard")

    # NEW: scarcity weighting parameter
    ap.add_argument("--scarcity_alpha", type=float, default=0.0, help="sponsored scarcity weighting α (0..~0.2). 0 disables the feature.")

    ap.add_argument("--min_score", type=float, default=None, help="optional filter: drop elig rows with score < min_score")
    ap.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="logging level")
    cfg = ap.parse_args()
    
    # Set log level from command line
    logging.getLogger().setLevel(getattr(logging, cfg.log_level))
    
    try:
        main(cfg)
    except Exception:
        logger.exception("Unhandled error during allocation run")
        sys.exit(1)


if __name__ == "__main__":
    cli()
