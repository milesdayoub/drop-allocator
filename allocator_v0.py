#!/usr/bin/env python3
"""
Claim – Offline Drop Allocator (POC-friendly)
──────────────────────────────────────────────────────────────────────────────
• Unsponsored contracts: if is_sponsored in {False, 0} and cap_face <= 0,
  assign synthetic capacity via --unsponsored_cap (default 10_000)
• Solver order selectable: --solver both|pulp|or  (default: both)
• If first solver returns empty, we try the next, then Greedy
• No hard failure on partial fills (default). Use --strict to enforce exactly k.
• Prints per-user offer distribution (3 / 2 / 1 / 0), objective, and timings.
"""

from __future__ import annotations
import argparse, sys, time, textwrap, math
import pandas as pd, numpy as np

# ── optional solvers ─────────────────────────────────────────────────────────
try:
    import pulp  # type: ignore
    HAVE_PULP = True
except Exception:
    HAVE_PULP = False

try:
    from ortools.sat.python import cp_model  # type: ignore
    HAVE_OR = True
except Exception:
    HAVE_OR = False

# Reasonable guard to avoid accidental 10-hour CP-SAT runs on laptops.
# Eligibility rows ≈ number of boolean vars.
MAX_CP_SAT_VARS = 6_000_000


# ╭──────────────────────────── helpers ────────────────────────────╮
def _to_int(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def load_inputs(caps_csv: str,
                elig_csv: str,
                unsponsored_cap: int,
                top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # contracts / caps
    caps = (pd.read_csv(caps_csv, dtype={'contract_address': str})
              .rename(columns=str.strip))

    # be robust to 0/1 or True/False in is_sponsored
    if 'is_sponsored' not in caps.columns:
        raise ValueError("caps missing {'is_sponsored'}")
    caps['cap_face'] = _to_int(caps['cap_face'])
    unspon_mask = (caps['is_sponsored'].isin([False, 0, '0', 'false', 'False', 'f', 'F']))
    zero_or_neg = caps['cap_face'] <= 0
    caps.loc[unspon_mask & zero_or_neg, 'cap_face'] = int(unsponsored_cap)

    # drop zero/negative capacity and dups
    caps = (caps[caps.cap_face > 0]
              .drop_duplicates('contract_address')
              .reset_index(drop=True))

    # eligibility pairs
    elig = (pd.read_csv(elig_csv, dtype={'contract_address': str, 'user_id': str})
              .rename(columns=str.strip)
              .drop_duplicates(['user_id', 'contract_address']))
    need_caps = {'contract_address', 'cap_face', 'is_sponsored'}
    need_elig = {'user_id', 'contract_address', 'score'}
    if m := (need_caps - set(caps.columns)):  raise ValueError(f"caps missing {m}")
    if m := (need_elig - set(elig.columns)):  raise ValueError(f"elig missing {m}")

    # keep only contracts that remain after pruning
    elig = elig[elig.contract_address.isin(caps.contract_address)]

    # optional top-N per user (OFF by default so we don’t re-trim your CSV)
    if top_n and top_n > 0:
        elig = (elig.sort_values(['user_id', 'score'], ascending=[True, False])
                    .groupby('user_id', sort=False)
                    .head(top_n)
                    .reset_index(drop=True))

    return caps.reset_index(drop=True), elig.reset_index(drop=True)


def summarize(df_assign: pd.DataFrame, elig: pd.DataFrame, k: int) -> dict:
    """Return offer count distribution per user (0..k) + some rates."""
    users = pd.Index(elig['user_id'].unique())
    per_user = (df_assign.groupby('user_id').size()
                .reindex(users, fill_value=0))
    dist = per_user.value_counts().reindex(range(0, k+1), fill_value=0).to_dict()
    fill_rate = per_user.sum() / (len(users) * k) if len(users) and k else 0.0
    return {
        'n_users': int(len(users)),
        'dist': {int(i): int(dist[i]) for i in dist},
        'fill_rate': float(fill_rate),
    }


def print_summary(df_assign: pd.DataFrame, caps: pd.DataFrame,
                  elig: pd.DataFrame, k: int,
                  label: str, t_sec: float):
    used = df_assign.contract_address.value_counts()
    cap = caps.set_index('contract_address').cap_face.reindex(used.index).fillna(0)
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


# ╰────────────────────────────────────────────────────────────────────────────╯


# ╭────────────────────────── algorithms ───────────────────────────╮
def greedy(caps, elig, k, seed=42):
    rng = np.random.default_rng(seed)  # reserved for tie breaks if needed later
    remaining = caps.set_index('contract_address').cap_face.to_dict()
    chosen = []
    for uid, grp in elig.groupby('user_id', sort=False):
        grp = grp.sort_values(['score', 'contract_address'], ascending=[False, True])
        taken = 0
        for _, row in grp.iterrows():
            c = row.contract_address
            if remaining.get(c, 0) > 0:
                chosen.append(row)
                remaining[c] -= 1
                taken += 1
                if taken == k:
                    break
    return pd.DataFrame(chosen, columns=elig.columns)


def ilp_pulp(caps, elig, k, timeout, cbc_verbose=False):
    if not HAVE_PULP:
        return None, 0.0
    t0 = time.time()

    prob = pulp.LpProblem("drop", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", elig.index, cat="Binary")
    # objective
    prob += pulp.lpSum(elig.loc[i, 'score'] * x[i] for i in elig.index)
    # constraints
    for uid, idx in elig.groupby('user_id').groups.items():
        prob += pulp.lpSum(x[i] for i in idx) <= k
    cap = caps.set_index('contract_address').cap_face.to_dict()
    for caddr, idx in elig.groupby('contract_address').groups.items():
        prob += pulp.lpSum(x[i] for i in idx) <= int(cap[caddr])

    solver = pulp.PULP_CBC_CMD(msg=cbc_verbose, timeLimit=int(timeout))
    prob.solve(solver)
    t = time.time() - t0

    status = pulp.LpStatus.get(prob.status, "Unknown")
    print(f"→ PuLP  ILP ……status={status}  time={t:.1f}s  objective={pulp.value(prob.objective):.6f}")
    if status not in ("Optimal", "Feasible"):
        return None, t

    pick = [i for i in elig.index if x[i].value() > 0.5]
    return elig.loc[pick], t


def ilp_ortools(caps, elig, k, timeout, or_workers=8, log_progress=False):
    if (not HAVE_OR) or (len(elig) > MAX_CP_SAT_VARS):
        return None, 0.0
    t0 = time.time()

    m = cp_model.CpModel()
    uid_cat = pd.Categorical(elig.user_id)
    cid_cat = pd.Categorical(elig.contract_address)
    x = [m.NewBoolVar(f"x{i}") for i in range(len(elig))]

    # per-user limit
    for ucode in range(len(uid_cat.categories)):
        idx = np.where(uid_cat.codes == ucode)[0]
        m.Add(sum(x[i] for i in idx) <= k)

    # per-contract limit
    cap_vec = caps.set_index('contract_address').cap_face
    for ccode in range(len(cid_cat.categories)):
        idx = np.where(cid_cat.codes == ccode)[0]
        addr = cid_cat.categories[ccode]
        m.Add(sum(x[i] for i in idx) <= int(cap_vec[addr]))

    # objective
    scores = (elig.score * 1_000_000).astype(int).to_numpy()
    m.Maximize(sum(scores[i] * x[i] for i in range(len(elig))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout)
    solver.parameters.num_search_workers = max(1, int(or_workers))
    if log_progress:
        solver.parameters.log_search_progress = True

    status = solver.Solve(m)
    t = time.time() - t0

    status_str = {cp_model.OPTIMAL: "OPTIMAL",
                  cp_model.FEASIBLE: "FEASIBLE"}.get(status, str(status))
    obj = solver.ObjectiveValue() / 1_000_000.0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float('nan')
    print(f"→ OR-Tools ILP …status={status_str}  time={t:.1f}s  objective={obj:.6f}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, t

    mask = np.fromiter((solver.Value(xi) for xi in x), bool, len(x))
    return elig[mask], t
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────────── main ───────────────────────────────╮
def main(cfg):
    t_start = time.time()
    caps, elig = load_inputs(cfg.caps, cfg.elig, cfg.unsponsored_cap, cfg.top_n)
    print(f"caps: {len(caps):,}   elig pairs: {len(elig):,}")

    df = None
    t_pulp = t_or = 0.0

    if cfg.solver in ("pulp", "both"):
        df, t_pulp = ilp_pulp(caps, elig, cfg.k, cfg.timeout, cfg.cbc_verbose)
    if (df is None or df.empty) and (cfg.solver in ("or", "both")):
        df, t_or = ilp_ortools(caps, elig, cfg.k, cfg.timeout,
                               or_workers=cfg.or_workers,
                               log_progress=cfg.or_log)

    if df is None or df.empty:
        print("→ Greedy fallback")
        df = greedy(caps, elig, cfg.k, cfg.rng)
        label = "Greedy"
        t_used = time.time() - t_start
    else:
        label = "PuLP" if t_pulp and (t_or == 0.0 or not (df is None or df.empty)) else "OR-Tools"
        t_used = (t_pulp if label == "PuLP" else t_or)

    # Always summarize; only enforce exactly k if --strict
    print_summary(df, caps, elig, cfg.k, label, t_used)

    # strict mode (optional)
    if cfg.strict:
        per_user = (df.groupby('user_id').size()
                      .reindex(elig['user_id'].unique(), fill_value=0))
        if (per_user != cfg.k).any():
            dist = per_user.value_counts().reindex(range(0, cfg.k+1), fill_value=0).to_dict()
            raise ValueError(f"Strict mode: some users not at k. Distribution={dist}")

    df.to_csv(cfg.out, index=False)

    t_total = time.time() - t_start
    print(f"✅ wrote {cfg.out}   (total wall time {t_total:.1f}s)")
# ╰─────────────────────────────────────────────────────────────────────────────╯


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--elig", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("-k", type=int, default=3, help="offers per user")
    ap.add_argument("--top_n", type=int, default=0,
                    help="keep N best contracts per user before solving (0 = no limit)")
    ap.add_argument("--unsponsored_cap", type=int, default=10_000,
                    help="synthetic capacity for unsponsored rows with <=0 cap")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="seconds per ILP solver (each)")
    ap.add_argument("--rng", type=int, default=42)
    ap.add_argument("--solver", choices=("both","pulp","or"), default="both",
                    help="which solver(s) to try")
    ap.add_argument("--cbc_verbose", action="store_true",
                    help="print CBC progress (PuLP)")
    ap.add_argument("--or_workers", type=int, default=8,
                    help="OR-Tools: number of search workers (threads)")
    ap.add_argument("--or_log", action="store_true",
                    help="OR-Tools: log search progress")
    ap.add_argument("--strict", action="store_true",
                    help="enforce exactly k offers per user (otherwise just report)")
    cfg = ap.parse_args()
    try:
        main(cfg)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
