/* ─── Timed Drop 2122 ── window: 2025-08-14 15:00 → 2025-08-15 07:00 UTC ── */
/* Knobs:
   - z: one-sided z for Wilson UCB on new-claim redemption (1.96=97.5%, 1.645=95%, 1.28=90%)
   - floor_r: floor redemption for new claims (prevents divide-by-zero / runaway caps)
   - overage_mult: multiply sponsored budget_left by this factor (1.0=no overage; 1.5–2.0 suggested)
   - no_hist_p_point: point estimate used to reserve outstanding when no history (ultra-conservative? choose 0.20 or 0.10)
   - no_hist_p_ucb: UCB used for new claims when no history (e.g., 0.25; keep it ≥ no_hist_p_point)
*/

WITH params AS (
  SELECT
    TIMESTAMPTZ '2025-08-21 15:00:00+00' AS win_start,
    TIMESTAMPTZ '2025-08-22 07:00:00+00' AS win_end,
    2122::int                           AS drop_id,
    1.28::numeric                       AS z,                -- loosen from 1.645→1.28 for more capacity
    0.03::numeric                       AS floor_r,          -- floor on new-claim redemption
    1.50::numeric                       AS overage_mult,     -- sponsored overage (1.0 = off)
    0.20::numeric                       AS no_hist_p_point,  -- for reserving OUTSTANDING when no history
    0.25::numeric                       AS no_hist_p_ucb     -- for NEW claims when no history
),
scope AS (                        -- contracts visible in this window and not excluded by this drop
    SELECT
        c.address                          AS contract_address,
        c.brand_id,
        c.is_sponsored,
        /* face we reserve against; prefer price when set, otherwise max_value */
        COALESCE(NULLIF(c.price,0), c.max_value, 0)::numeric AS face_value
    FROM   contract c
    CROSS  JOIN params p
    WHERE  c.is_drop = TRUE
      AND  c.start_datetime <= p.win_end
      AND  c.end_datetime   >= p.win_start
      AND  c.address NOT IN (
            SELECT contract_address
            FROM   timed_drop_excluded_contract
            WHERE  timed_drop_id = p.drop_id
      )
),
spent AS (                         -- dollars already paid
    SELECT contract_address,
           SUM(amount)::numeric AS dollars_spent
    FROM   payout
    WHERE  settled_datetime IS NOT NULL
      AND  contract_address IN (SELECT contract_address FROM scope)
    GROUP  BY 1
),
outstanding AS (                   -- live, unredeemed claims
    SELECT contract_address,
           COUNT(*) AS coupons_live
    FROM   contract_assignment
    WHERE  contract_address IN (SELECT contract_address FROM scope)
      AND  claimed_datetime  IS NULL
      AND  rejected_datetime IS NULL
      AND (expiration_datetime IS NULL OR expiration_datetime > now())
    GROUP  BY 1
),
hist AS (                          -- 180-day assignment ↔ redemption history
    SELECT ca.contract_address,
           COUNT(*)                                   AS assigns_180d,
           SUM(CASE WHEN cv.token_pk IS NOT NULL THEN 1 ELSE 0 END) AS redeems_180d
    FROM   contract_assignment ca
    LEFT   JOIN conversion cv ON cv.token_pk = ca.token_pk
    WHERE  ca.contract_address IN (SELECT contract_address FROM scope)
      AND  ca.assigned_datetime >= now() - INTERVAL '180 days'
    GROUP  BY 1
),
redeem_rates AS (                  -- point estimate + UCB + floor handling
    SELECT
      s.contract_address,
      s.face_value,
      s.is_sponsored,
      COALESCE(h.assigns_180d,0) AS n,
      COALESCE(h.redeems_180d,0) AS r,
      /* Laplace-smoothed point estimate: used to reserve OUTSTANDING cost */
      CASE
        WHEN COALESCE(h.assigns_180d,0) = 0
          THEN (SELECT no_hist_p_point FROM params)
        ELSE (h.redeems_180d + 1.0)::numeric / (h.assigns_180d + 2.0)
      END AS p_point,
      /* One-sided Wilson UCB: used for NEW claims (knobbed by z) */
      CASE
        WHEN COALESCE(h.assigns_180d,0) = 0
          THEN (SELECT no_hist_p_ucb FROM params)
        ELSE LEAST(
               1.0,
               (h.redeems_180d + 1.0)::numeric / (h.assigns_180d + 2.0)
               + (SELECT z FROM params) *
                 sqrt(
                   ((h.redeems_180d + 1.0)::numeric
                    * (h.assigns_180d - h.redeems_180d + 1.0))
                   / ( ((h.assigns_180d + 2.0)^2)::numeric * (h.assigns_180d + 3.0) )
                 )
             )
      END AS p_ucb
    FROM scope s
    LEFT JOIN hist h USING (contract_address)
),
budget_frame AS (
    SELECT
      s.contract_address,
      s.brand_id,
      s.is_sponsored,
      rr.face_value,
      /* nominal budget left before overage */
      GREATEST(c.budget::numeric - COALESCE(sp.dollars_spent,0), 0)::numeric AS budget_left_base,
      COALESCE(o.coupons_live,0) AS coupons_live,
      rr.p_point,
      GREATEST(rr.p_ucb, (SELECT floor_r FROM params)) AS p_eff_new  -- enforce floor for NEW claims only
    FROM scope s
    JOIN contract c           ON c.address = s.contract_address
    LEFT JOIN spent sp        ON sp.contract_address = s.contract_address
    LEFT JOIN outstanding o   ON o.contract_address  = s.contract_address
    LEFT JOIN redeem_rates rr ON rr.contract_address = s.contract_address
),
capacities AS (
    SELECT
      bf.contract_address,
      bf.brand_id,
      bf.is_sponsored,
      bf.face_value,
      /* apply sponsored overage multiplier */
      CASE WHEN bf.is_sponsored
           THEN (SELECT overage_mult FROM params) * bf.budget_left_base
           ELSE bf.budget_left_base
      END AS budget_left_eff,
      bf.coupons_live,
      bf.p_point,
      bf.p_eff_new
    FROM budget_frame bf
)
SELECT
    c.contract_address,
    c.brand_id,
    c.is_sponsored,
    c.face_value,
    /* report budgets (post-overage for sponsored) */
    (SELECT budget_left_base FROM budget_frame b WHERE b.contract_address = c.contract_address) AS budget_total, -- original contract.budget
    c.budget_left_eff AS budget_left,
    c.coupons_live     AS outstanding,
    COALESCE(h.assigns_180d,0) AS assigns_180d,
    COALESCE(h.redeems_180d,0) AS redeems_180d,
          /* effective UCB used for NEW claims (after floor) */
    c.p_eff_new        AS p_upper,
          /* safe additional claims we can mint now:
       remaining dollars after reserving outstanding at p_point,
       divided by face_value × p_eff_new
    */
    FLOOR(
      GREATEST(
        c.budget_left_eff
        - (c.coupons_live * c.face_value * c.p_point),   -- reserve for OUTSTANDING at point estimate
        0
      )
      / NULLIF(c.face_value * c.p_eff_new, 0)
    )::int AS cap_face
FROM capacities c
LEFT JOIN hist h USING (contract_address)
ORDER  BY c.contract_address;