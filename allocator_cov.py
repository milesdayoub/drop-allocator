#!/usr/bin/env python3
"""
Claim – Drop Allocator with coverage encouragement (POC)
• Sponsored caps already include redemption projection (cap_face from PG).
• Unsponsored synthetic cap via --unsponsored_cap (no redemption modeling).
• Adds coverage bonus per user slot (1st/2nd/3rd) to reduce 0/1-offer users.
• Solver order selectable: --solver both|pulp|or.
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

MAX_CP_SAT_VARS = 6_000_000  # guard

def _to_int(s): return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def load_inputs(caps_csv, elig_csv, unsponsored_cap, top_n):
    caps = (pd.read_csv(caps_csv, dtype={'contract_address': str})
              .rename(columns=str.strip))
    if 'is_sponsored' not in caps.columns:
        raise ValueError("caps missing {'is_sponsored'}")
    caps['cap_face'] = _to_int(caps['cap_face'])

    unspon_mask = caps['is_sponsored'].isin([False, 0, '0', 'false', 'False', 'f', 'F'])
    caps.loc[unspon_mask & (caps['cap_face'] <= 0), 'cap_face'] = int(unsponsored_cap)

    caps = (caps[caps.cap_face > 0]
              .drop_duplicates('contract_address')
              .reset_index(drop=True))

    elig = (pd.read_csv(elig_csv, dtype={'contract_address': str, 'user_id': str})
              .rename(columns=str.strip)
              .drop_duplicates(['user_id','contract_address']))

    need_caps = {'contract_address','cap_face','is_sponsored'}
    need_elig = {'user_id','contract_address','score'}
    if (m := need_caps - set(caps.columns)):  raise ValueError(f"caps missing {m}")
    if (m := need_elig - set(elig.columns)):  raise ValueError(f"elig missing {m}")

    # Only keep contracts that have positive cap after the above
    elig = elig[elig.contract_address.isin(caps.contract_address)]

    # Optional per-user Top-N before solve (0 = keep all)
    if top_n and top_n > 0:
        elig = (elig.sort_values(['user_id','score'], ascending=[True,False])
                    .groupby('user_id', sort=False)
                    .head(top_n)
                    .reset_index(drop=True))
    return caps.reset_index(drop=True), elig.reset_index(drop=True)

def summarize(df_assign: pd.DataFrame, elig: pd.DataFrame, k: int):
    users = pd.Index(elig['user_id'].unique())
    per_user = df_assign.groupby('user_id').size().reindex(users, fill_value=0)
    dist = per_user.value_counts().reindex(range(0,k+1), fill_value=0).to_dict()
    fill = per_user.sum() / (len(users) * k) if len(users) and k else 0.0
    return {'n_users': int(len(users)),
            'dist': {int(i): int(dist[i]) for i in dist},
            'fill_rate': float(fill)}

def print_summary(df_assign, caps, elig, k, label, t_sec):
    used = df_assign.contract_address.value_counts()
    cap  = caps.set_index('contract_address').cap_face.reindex(used.index).fillna(0)
    cap_viol = used[used > cap]
    stats = summarize(df_assign, elig, k)
    obj = df_assign['score'].sum()
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
        grp = grp.sort_values(['score','contract_address'], ascending=[False,True])
        take = 0
        for _, row in grp.iterrows():
            c = row.contract_address
            if remaining.get(c,0) > 0:
                chosen.append(row); remaining[c] -= 1
                take += 1
                if take == k: break
    return pd.DataFrame(chosen, columns=elig.columns)

def _parse_cov_w(cov_w_str: str, k: int):
    if not cov_w_str: return [0.0]*k
    parts = [float(x.strip()) for x in cov_w_str.split(',') if x.strip()]
    if len(parts) < k: parts += [0.0]*(k - len(parts))
    return parts[:k]

# ───────────── PuLP (with coverage bonus) ─────────────
def ilp_pulp(caps, elig, k, timeout, cov_w_str):
    if not HAVE_PULP: return None, 0.0
    t0 = time.time()

    cov_w = _parse_cov_w(cov_w_str, k)  # e.g., [0.0003,0.0006,0.001]
    prob = pulp.LpProblem("drop", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in elig.index}

    # pre-sum indices by user and by contract
    by_user = elig.groupby('user_id').groups
    by_c    = elig.groupby('contract_address').groups

    # coverage slot variables z_{u,t}, t = 1..k (1st..kth offer)
    z = {}
    for u in by_user.keys():
        for t in range(1, k+1):
            z[(u,t)] = pulp.LpVariable(f"z_{u}_{t}", cat="Binary")

    # objective = score + sum_u sum_t cov_w[t-1] * z_{u,t}
    prob += pulp.lpSum(elig.loc[i,'score'] * x[i] for i in elig.index) + \
            pulp.lpSum(cov_w[t-1]*z[(u,t)] for u in by_user.keys() for t in range(1,k+1))

    # per-user ≤ k
    for u, idx in by_user.items():
        prob += pulp.lpSum(x[i] for i in idx) <= k

    # coverage linking: sum_i x_{u,i} >= t * z_{u,t}
    for u, idx in by_user.items():
        su = pulp.lpSum(x[i] for i in idx)
        for t in range(1, k+1):
            prob += su - t * z[(u,t)] >= 0

    # per-contract ≤ cap
    cap = caps.set_index('contract_address').cap_face.to_dict()
    for caddr, idx in by_c.items():
        prob += pulp.lpSum(x[i] for i in idx) <= int(cap[caddr])

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(timeout))
    prob.solve(solver)
    t = time.time() - t0
    status = pulp.LpStatus.get(prob.status, "Unknown")
    obj = float(pulp.value(prob.objective)) if status in ("Optimal","Feasible") else float('nan')
    print(f"→ PuLP  ILP ……status={status}  time={t:.1f}s  objective={obj:.6f}")
    if status not in ("Optimal","Feasible"): return None, t

    pick = [i for i in elig.index if x[i].value() > 0.5]
    return elig.loc[pick], t

# ───────────── OR-Tools CP-SAT (with coverage bonus) ─────────────
def ilp_ortools(caps, elig, k, timeout, or_workers, or_log, cov_w_str):
    if (not HAVE_OR) or (len(elig) > MAX_CP_SAT_VARS): return None, 0.0
    t0 = time.time()
    cov_w = _parse_cov_w(cov_w_str, k)

    m = cp_model.CpModel()
    uid_cat = pd.Categorical(elig.user_id)
    cid_cat = pd.Categorical(elig.contract_address)

    # x vars
    x = [m.NewBoolVar(f"x{i}") for i in range(len(elig))]

    # prebuild index lists
    idx_by_user = {uc: np.where(uid_cat.codes == uc)[0] for uc in range(len(uid_cat.categories))}
    idx_by_c    = {cc: np.where(cid_cat.codes == cc)[0] for cc in range(len(cid_cat.categories))}

    # per-user ≤ k
    for uc, idx in idx_by_user.items():
        m.Add(sum(x[i] for i in idx) <= k)

    # per-contract ≤ cap
    cap_vec = caps.set_index('contract_address').cap_face
    for cc, idx in idx_by_c.items():
        addr = cid_cat.categories[cc]
        m.Add(sum(x[i] for i in idx) <= int(cap_vec[addr]))

    # coverage z_{u,t} and linking: sum_x(u) >= t * z_{u,t}
    z = {}
    cov_terms = []
    for uc, idx in idx_by_user.items():
        su = sum(x[i] for i in idx)
        for t in range(1, k+1):
            zz = m.NewBoolVar(f"z_{uc}_{t}")
            m.Add(su - t * zz >= 0)
            z[(uc,t)] = zz
            if cov_w[t-1] != 0.0:
                cov_terms.append(int(cov_w[t-1] * 1_000_000) * zz)  # scale to int

    # objective: score + coverage bonus
    scores = (elig.score * 1_000_000).astype(int).to_numpy()
    m.Maximize(sum(scores[i]*x[i] for i in range(len(elig))) + sum(cov_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout)
    solver.parameters.num_search_workers = max(1, int(or_workers))
    solver.parameters.log_search_progress = bool(or_log)

    status = solver.Solve(m)
    t = time.time() - t0
    status_str = {cp_model.OPTIMAL:"OPTIMAL", cp_model.FEASIBLE:"FEASIBLE"}.get(status, str(status))
    obj = solver.ObjectiveValue()/1_000_000.0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float('nan')
    print(f"→ OR-Tools ILP …status={status_str}  time={t:.1f}s  objective={obj:.6f}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None, t

    mask = np.fromiter((solver.Value(xi) for xi in x), bool, len(x))
    return elig[mask], t

def main(cfg):
    t0 = time.time()
    caps, elig = load_inputs(cfg.caps, cfg.elig, cfg.unsponsored_cap, cfg.top_n)
    print(f"caps: {len(caps):,}   elig pairs: {len(elig):,}")

    df = None; t_used = 0.0
    if cfg.solver in ("pulp","both"):
        df, t_used = ilp_pulp(caps, elig, cfg.k, cfg.timeout, cfg.cov_w)
    if (df is None or df.empty) and (cfg.solver in ("or","both")):
        df, t_used = ilp_ortools(caps, elig, cfg.k, cfg.timeout, cfg.or_workers, cfg.or_log, cfg.cov_w)
    if df is None or df.empty:
        print("→ Greedy fallback")
        df = greedy(caps, elig, cfg.k, cfg.rng)
        label = "Greedy"
    else:
        label = "PuLP" if cfg.solver=="pulp" or (cfg.solver=="both" and t_used>0) else "OR-Tools"

    print_summary(df, caps, elig, cfg.k, label, t_used)
    df.to_csv(cfg.out, index=False)
    print(f"✅ wrote {cfg.out}   (total wall time {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--elig", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("-k", type=int, default=3, help="offers per user")
    ap.add_argument("--top_n", type=int, default=0, help="pre-trim per user (0=off)")
    ap.add_argument("--unsponsored_cap", type=int, default=10_000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--rng", type=int, default=42)
    ap.add_argument("--solver", choices=("both","pulp","or"), default="both")
    ap.add_argument("--or_workers", type=int, default=8)
    ap.add_argument("--or_log", action="store_true")
    ap.add_argument("--cov_w", default="0.0003,0.0006,0.001",
                    help="coverage weights for 1st,2nd,3rd offers per user")
    cfg = ap.parse_args()
    try:
        main(cfg)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr); sys.exit(1)
