/* ─── Timed Drop 2122 ── window: 2025-08-14 15:00 → 2025-08-15 07:00 UTC ── */
/* Knobs (global defaults; per-contract overrides come below):
   - z: one-sided z for Wilson UCB on new-claim redemption (lower → higher caps)
- floor_r: floor redemption for new claims
   - overage_mult: default sponsored overage (1.0 = off)
*/

WITH params AS (
  SELECT
    TIMESTAMPTZ '2025-08-21 15:00:00+00' AS win_start,
    TIMESTAMPTZ '2025-08-22 07:00:00+00' AS win_end,
    2122::int                           AS drop_id,
    1.28::numeric                       AS z,
    0.03::numeric                       AS floor_r,
    1.50::numeric                       AS overage_mult
),

/* 🔸 Per-contract cap boost for the 12 fully-utilized SPONSORED contracts only.
   For each: you can set either/both:
   - overage_mult_override (e.g., 2.0 or 2.5)
   - extra_budget_usd (fixed dollar top-up)
*/
overrides(contract_address, overage_mult_override, extra_budget_usd) AS (
  VALUES
    -- TODO: paste YOUR 12 addresses below. Examples shown as placeholders:
    ('0x00000000000000000000000000000000000596', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000601', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000602', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000603', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000604', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000605', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000608', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000611', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000612', 2.50::numeric, NULL::numeric),
    ('0x00000000000000000000000000000000000617', 2.50::numeric, NULL::numeric)
    -- (add the remaining 2 here)
),

scope AS (
  SELECT
      c.address AS contract_address,
      c.brand_id,
      c.is_sponsored,
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

spent AS (
  SELECT contract_address, SUM(amount)::numeric AS dollars_spent
  FROM   payout
  WHERE  settled_datetime IS NOT NULL
    AND  contract_address IN (SELECT contract_address FROM scope)
  GROUP  BY 1
),

outstanding AS (
  SELECT contract_address, COUNT(*) AS coupons_live
  FROM   contract_assignment
  WHERE  contract_address IN (SELECT contract_address FROM scope)
    AND  claimed_datetime  IS NULL
    AND  rejected_datetime IS NULL
    AND (expiration_datetime IS NULL OR expiration_datetime > now())
  GROUP  BY 1
),

hist AS (
  SELECT ca.contract_address,
         COUNT(*) AS assigns_180d,
         SUM((cv.token_pk IS NOT NULL)::int) AS redeems_180d
  FROM   contract_assignment ca
  LEFT   JOIN conversion cv ON cv.token_pk = ca.token_pk
  WHERE  ca.contract_address IN (SELECT contract_address FROM scope)
    AND  ca.assigned_datetime >= now() - INTERVAL '180 days'
  GROUP  BY 1
),

redeem_rates AS (
  SELECT
    s.contract_address,
    s.face_value,
    s.is_sponsored,
    COALESCE(h.assigns_180d,0) AS n,
    COALESCE(h.redeems_180d,0) AS r,
    /* Laplace-smoothed point estimate for OUTSTANDING reservation */
    CASE WHEN COALESCE(h.assigns_180d,0) = 0
         THEN 0.20::numeric
         ELSE (h.redeems_180d + 1.0)::numeric / (h.assigns_180d + 2.0)
    END AS p_point,
    /* One-sided Wilson UCB for NEW claims (using params.z), floored later */
    CASE WHEN COALESCE(h.assigns_180d,0) = 0
         THEN 0.25::numeric
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
    GREATEST(c.budget::numeric - COALESCE(sp.dollars_spent,0), 0)::numeric AS budget_left_base,
    COALESCE(o.coupons_live,0) AS coupons_live,
    rr.p_point,
    GREATEST(rr.p_ucb, (SELECT floor_r FROM params)) AS p_eff_new
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
    /* 🔸 Apply per-contract override if present, else default */
    CASE
      WHEN bf.is_sponsored THEN
        COALESCE(ov.overage_mult_override, (SELECT overage_mult FROM params)) * bf.budget_left_base
        + COALESCE(ov.extra_budget_usd, 0)
      ELSE
        bf.budget_left_base
    END AS budget_left_eff,
    bf.coupons_live,
    bf.p_point,
    bf.p_eff_new
  FROM budget_frame bf
  LEFT JOIN overrides ov ON ov.contract_address = bf.contract_address
)

SELECT
    c.contract_address,
    c.brand_id,
    c.is_sponsored,
    c.face_value,
    /* report budgets (post-overage for sponsored) */
    (SELECT budget_left_base FROM budget_frame b WHERE b.contract_address = c.contract_address) AS budget_total,
    c.budget_left_eff AS budget_left,
    c.coupons_live     AS outstanding,
    COALESCE(h.assigns_180d,0) AS assigns_180d,
    COALESCE(h.redeems_180d,0) AS redeems_180d,
    c.p_eff_new        AS p_upper,
    FLOOR(
      GREATEST(
        c.budget_left_eff
        - (c.coupons_live * c.face_value * c.p_point),
        0
      ) / NULLIF(c.face_value * c.p_eff_new, 0)
    )::int AS cap_face
FROM capacities c
LEFT JOIN hist h USING (contract_address)
ORDER  BY c.contract_address;
