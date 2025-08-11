/* ─── Timed Drop 2121 ── window: 2025‑08‑14 15:00 → 2025‑08‑15 07:00 UTC ── */

WITH
scope AS (                        -- contracts visible in this window
    SELECT
        c.address                          AS contract_address,
        c.brand_id,
        c.is_sponsored,
        COALESCE(NULLIF(c.price,0), c.max_value, 0)::numeric AS face_value
    FROM   contract c
    WHERE  c.is_drop = TRUE
      AND  c.start_datetime <= TIMESTAMPTZ '2025-08-15 07:00:00+00'
      AND  c.end_datetime   >= TIMESTAMPTZ '2025-08-14 15:00:00+00'
      AND  c.address NOT IN (
            SELECT contract_address
            FROM   timed_drop_excluded_contract
            WHERE  timed_drop_id = 2121
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
outstanding AS (                   -- live, unredeemed coupons
    SELECT contract_address,
           COUNT(*) AS coupons_live
    FROM   contract_assignment
    WHERE  contract_address IN (SELECT contract_address FROM scope)
      AND  claimed_datetime  IS NULL
      AND  rejected_datetime IS NULL
      AND (expiration_datetime IS NULL OR expiration_datetime > now())
    GROUP  BY 1
),
hist AS (                          -- 180‑day assignment ↔ redemption history
    SELECT ca.contract_address,
           COUNT(*)                                   AS assigns_180d,
           SUM(CASE WHEN cv.token_pk IS NOT NULL THEN 1 ELSE 0 END) AS redeems_180d
    FROM   contract_assignment ca
    LEFT   JOIN conversion cv ON cv.token_pk = ca.token_pk
    WHERE  ca.contract_address IN (SELECT contract_address FROM scope)
      AND  ca.assigned_datetime >= now() - INTERVAL '180 days'
    GROUP  BY 1
)
SELECT
    s.contract_address,
    s.brand_id,
    s.is_sponsored,
    s.face_value,
    c.budget::numeric                                   AS budget_total,
    GREATEST(c.budget - COALESCE(sp.dollars_spent,0),0) AS budget_left,
    COALESCE(o.coupons_live,0)                          AS outstanding,
    COALESCE(h.assigns_180d,0)                          AS assigns_180d,
    COALESCE(h.redeems_180d,0)                          AS redeems_180d,
    /* 97.5 % upper bound on redemption‑rate
       (Wilson/Laplace approximation → avoids beta_inv() which may be absent) */
    CASE
      WHEN COALESCE(h.assigns_180d,0) = 0 THEN 0.97
      ELSE
        LEAST(1,
              (h.redeems_180d + 1.0)/(h.assigns_180d + 2.0) +
              1.96*sqrt( (h.redeems_180d + 1.0)*(h.assigns_180d - h.redeems_180d + 1.0)
                          / ((h.assigns_180d + 2.0)^2*(h.assigns_180d + 3.0)) )
        )
    END                                               AS p_upper,
    /* safe amount of new coupons we can still print */
    FLOOR(
      GREATEST(c.budget - COALESCE(sp.dollars_spent,0)
               - COALESCE(o.coupons_live,0)*s.face_value, 0)
      / NULLIF(
          s.face_value *
          GREATEST(          -- enforce 5 % minimum so cap never explodes
            CASE
              WHEN COALESCE(h.assigns_180d,0) = 0 THEN 0.97 ELSE
                   LEAST(1,
                        (h.redeems_180d + 1.0)/(h.assigns_180d + 2.0) +
                        1.96*sqrt( (h.redeems_180d + 1.0)*(h.assigns_180d - h.redeems_180d + 1.0)
                                    / ((h.assigns_180d + 2.0)^2*(h.assigns_180d + 3.0)) )
                   )
            END,
            0.05
          ), 1
        )
    )::int                                            AS cap_face
FROM   scope            s
JOIN   contract         c   ON c.address = s.contract_address
LEFT   JOIN spent       sp  ON sp.contract_address = s.contract_address
LEFT   JOIN outstanding o   ON o.contract_address = s.contract_address
LEFT   JOIN hist        h   ON h.contract_address = s.contract_address
ORDER  BY s.contract_address;