/* ─── Timed Drop 2121 – window + knobs ─────────────────────────────────────────
   Inputs you can tweak:
     :win_start, :win_end        → drop visibility window
     :history_days               → lookback window for redemption history
     :z                          → one-sided Wilson z (risk knob for NEW claims)
     :floor_r                    → minimum assumed redemption (safety floor)
     :overage_mult               → sponsored budget overage multiplier
-------------------------------------------------------------------------------*/

WITH
params AS (
  SELECT
    TIMESTAMPTZ '2025-08-14 15:00:00+00' AS win_start,    -- :win_start
    TIMESTAMPTZ '2025-08-15 07:00:00+00' AS win_end,      -- :win_end
    180::int                           AS history_days,   -- :history_days
    1.645::numeric                     AS z,              -- :z  (1.645≈95%  1.96≈97.5%  1.28≈90%)
    0.05::numeric                      AS floor_r,        -- :floor_r (min assumed redemption)
    1.00::numeric                      AS overage_mult    -- :overage_mult (1.50 = +50% over)
),
scope AS (  -- contracts visible in this window, excluding explicit drop exclusions
  SELECT
    c.address        AS contract_address,
    c.brand_id,
    c.is_sponsored,
    COALESCE(NULLIF(c.price,0), c.max_value, 0)::numeric AS face_value,
    c.budget::numeric                                   AS budget_total
  FROM contract c
  CROSS JOIN params p
  WHERE c.is_drop = TRUE
    AND c.start_datetime <= p.win_start
    AND c.end_datetime   >= p.win_end
    AND c.address NOT IN (
      SELECT contract_address
      FROM   timed_drop_excluded_contract
      WHERE  timed_drop_id = 2121
    )
),
spent AS (  -- dollars already paid
  SELECT contract_address,
         SUM(amount)::numeric AS dollars_spent
  FROM payout
  WHERE settled_datetime IS NOT NULL
    AND contract_address IN (SELECT contract_address FROM scope)
  GROUP BY 1
),
outstanding AS (  -- live, unredeemed, unexpired assignments ("claims")
  SELECT contract_address,
         COUNT(*)::int AS coupons_live
  FROM   contract_assignment
  WHERE  contract_address IN (SELECT contract_address FROM scope)
    AND  claimed_datetime  IS NULL
    AND  rejected_datetime IS NULL
    AND (expiration_datetime IS NULL OR expiration_datetime > now())
  GROUP BY 1
),
hist AS (  -- last N days assignment ↔ redemption history
  SELECT
    ca.contract_address,
    COUNT(*)::int AS assigns_n,
    SUM(CASE WHEN cv.token_pk IS NOT NULL THEN 1 ELSE 0 END)::int AS redeems_n
  FROM contract_assignment ca
  LEFT JOIN conversion cv ON cv.token_pk = ca.token_pk
  CROSS JOIN params p
  WHERE ca.contract_address IN (SELECT contract_address FROM scope)
    AND ca.assigned_datetime >= now() - (p.history_days || ' days')::interval
  GROUP BY 1
),
rates AS (  -- point estimate & Wilson upper bound + floors
  SELECT
    h.contract_address,
    h.assigns_n,
    h.redeems_n,
    -- Laplace point estimate for EXPECTED redemption (used for outstanding reserve)
    COALESCE( (h.redeems_n + 1.0) / NULLIF(h.assigns_n + 2.0, 0), 0.10 )    AS r_point_raw,  -- 0.10 fallback if no history
    -- Wilson one-sided upper bound (add-2 smoothed form) for NEW claim risk
    LEAST(
      1.0,
      (h.redeems_n + 1.0) / (h.assigns_n + 2.0)
      + (SELECT z FROM params) * sqrt(
            ((h.redeems_n + 1.0) * (h.assigns_n - h.redeems_n + 1.0))
            / ((h.assigns_n + 2.0)^2 * (h.assigns_n + 3.0))
        )
    ) AS r_upper_raw
  FROM hist h
)
SELECT
  s.contract_address,
  s.brand_id,
  s.is_sponsored,
  s.face_value,
  s.budget_total,
  -- allow overage ONLY on sponsored budgets
  (CASE WHEN s.is_sponsored THEN s.budget_total * p.overage_mult ELSE s.budget_total END)
    - COALESCE(sp.dollars_spent,0)                                        AS budget_left_allowed,
  COALESCE(o.coupons_live,0)                                              AS outstanding,
  COALESCE(h.assigns_n,0)                                                 AS assigns_hist,
  COALESCE(h.redeems_n,0)                                                 AS redeems_hist,
  -- redemption knobs actually used
  GREATEST(COALESCE(r.r_point_raw, 0.10), p.floor_r)                       AS r_outstanding_used,  -- expected for existing claims
  GREATEST(COALESCE(r.r_upper_raw, 0.97), p.floor_r)                       AS r_newcoupon_used,    -- upper bound for new issues
-- SAFE capacity (new claims we can still print)
  FLOOR(
    GREATEST(
      (CASE WHEN s.is_sponsored THEN s.budget_total * p.overage_mult ELSE s.budget_total END)
      - COALESCE(sp.dollars_spent,0)
      - COALESCE(o.coupons_live,0) * s.face_value * GREATEST(COALESCE(r.r_point_raw,0.10), p.floor_r),
      0
    )
    / NULLIF(s.face_value * GREATEST(COALESCE(r.r_upper_raw,0.97), p.floor_r), 0)
  )::int AS cap_face
FROM scope s
LEFT JOIN spent       sp ON sp.contract_address = s.contract_address
LEFT JOIN outstanding o  ON o.contract_address  = s.contract_address
LEFT JOIN hist        h  ON h.contract_address  = s.contract_address
LEFT JOIN rates       r  ON r.contract_address  = s.contract_address
CROSS JOIN params p
ORDER BY s.contract_address;