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


def load_inputs(caps_csv, elig_csv, unsponsored_cap, top_n, min_score):
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

    elig = (pd.read_csv(elig_csv, dtype={'contract_address': str, 'user_id': str})
              .rename(columns=str.strip)
              .drop_duplicates(['user_id', 'contract_address']))

    need_caps = {'contract_address', 'cap_face', 'is_sponsored'}
    need_elig = {'user_id', 'contract_address', 'score'}
    if (m := need_caps - set(caps.columns)):
        raise ValueError(f"caps missing {m}")
    if (m := need_elig - set(elig.columns)):
        raise ValueError(f"elig missing {m}")

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
    return caps.reset_index(drop=True), elig.reset_index(drop=True)


def summarize(df_assign: pd.DataFrame, elig: pd.DataFrame, k: int):
    users = pd.Index(elig['user_id'].unique())
    per_user = df_assign.groupby('user_id').size().reindex(users, fill_value=0)
    dist = per_user.value_counts().reindex(range(0, k+1), fill_value=0).to_dict()
    fill = per_user.sum() / (len(users) * k) if len(users) and k else 0.0
    return {'n_users': int(len(users)),
            'dist': {int(i): int(dist[i]) for i in dist},
            'fill_rate': float(fill)}


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


def greedy(caps, elig, k, seed=42):
    rng = np.random.default_rng(seed)
    remaining = caps.set_index('contract_address').cap_face.to_dict()
    chosen = []
    for uid, grp in elig.groupby('user_id', sort=False):
        grp = grp.sort_values(['score', 'contract_address'], ascending=[False, True])
        take = 0
        for _, row in grp.iterrows():
            c = row.contract_address
            if remaining.get(c, 0) > 0:
                chosen.append(row); remaining[c] -= 1
                take += 1
                if take == k: break
    return pd.DataFrame(chosen, columns=elig.columns)


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

    # objective = score + sum_u sum_t cov_w[t-1] * z_{u,t}
    prob += pulp.lpSum(elig.loc[i, 'score'] * x[i] for i in elig.index) + \
            pulp.lpSum(cov_w[t-1] * z[(u, t)] for u in by_user.keys() for t in range(1, k+1))

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

    # objective terms
    scores = (elig.score * 1_000_000).astype(int).to_numpy()
    obj_terms = [scores[i] * x[i] for i in range(len(elig))]

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
        m.Maximize(LSum(obj_terms) + LSum(cov_terms))
    else:
        # shortfall mode: per-user shortfall penalty (lighter/faster)
        pen = int(float(shortfall_penalty) * 1_000_000)  # penalty per missing offer
        short_vars = []
        for uc, idx in idx_by_user.items():
            short_u = m.NewIntVar(0, k, f"short_{uc}")
            m.Add(LSum([x[i] for i in idx]) + short_u >= k)
            short_vars.append(short_u)
        m.Maximize(LSum(obj_terms) - pen * LSum(short_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout)
    solver.parameters.num_search_workers = max(1, int(or_workers))
    solver.parameters.log_search_progress = bool(or_log)

    # Warm start from greedy (optional)
    if warm_start:
        t_ws = time.time()
        g = greedy(caps, elig, k, seed=42)
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
    caps, elig = load_inputs(cfg.caps, cfg.elig, cfg.unsponsored_cap, cfg.top_n, cfg.min_score)
    print(f"caps: {len(caps):,}   elig pairs: {len(elig):,}")

    df = None; t_used = 0.0
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
        df = greedy(caps, elig, cfg.k, cfg.rng)
        label = "Greedy"
    else:
        label = "PuLP" if cfg.solver == "pulp" or (cfg.solver == "both" and t_used > 0 and df is not None) else "OR-Tools"

    print_summary(df, caps, elig, cfg.k, label, t_used)
    df.to_csv(cfg.out, index=False)
    print(f"✅ wrote {cfg.out}   (total wall time {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--elig", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=3, help="offers per user")
    ap.add_argument("--top_n", type=int, default=0, help="pre-trim per user (0=off)")
    ap.add_argument("--unsponsored_cap", type=int, default=10_000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--rng", type=int, default=42)
    ap.add_argument("--solver", choices=("both", "pulp", "or"), default="both")
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
