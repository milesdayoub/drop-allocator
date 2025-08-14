-- user_groups.csv
-- Inputs: :win_start (timestamptz), :win_end (timestamptz), :drop_id (int or NULL)

WITH eligible_users AS (
  SELECT DISTINCT uce.user_id
  FROM   public.user_contract_eligibility uce
  JOIN   public.contract c
         ON c.address = uce.contract_address
  WHERE  uce.is_eligible = TRUE
    AND  c.is_drop = TRUE
    AND  c.is_onboarding = FALSE
    -- active in window: any overlap with [win_start, win_end]
    AND  c.start_datetime <= '2025-08-21 15:00:00+00'
    AND  c.end_datetime   >= '2025-08-22 07:00:00+00'
),
group_from_agg AS (
  SELECT ua.user_id, ua.primary_group_id AS agg_group_id
  FROM   public.user_agg ua
  JOIN   eligible_users eu USING (user_id)
  WHERE  ua.primary_group_id IS NOT NULL
),
group_from_map AS (
  SELECT mug.user_id, mug.group_id AS map_group_id
  FROM   public.map_user_group mug
  JOIN   eligible_users eu USING (user_id)
  WHERE  mug.is_primary = TRUE
),
chosen AS (
  -- prefer user_agg.primary_group_id, else map_user_group.is_primary
  SELECT
    eu.user_id,
    COALESCE(ga.agg_group_id, gm.map_group_id) AS group_id,
    ga.agg_group_id,
    gm.map_group_id
  FROM eligible_users eu
  LEFT JOIN group_from_agg ga ON ga.user_id = eu.user_id
  LEFT JOIN group_from_map gm ON gm.user_id = eu.user_id
)
SELECT t.user_id, t.group_id
FROM (
  -- deterministic single choice per user (prefer agg_group_id)
  SELECT
    c.user_id,
    c.group_id,
    ROW_NUMBER() OVER (
      PARTITION BY c.user_id
      ORDER BY
        CASE WHEN c.agg_group_id IS NOT NULL THEN 0 ELSE 1 END,
        c.group_id
    ) AS rn
  FROM chosen c
  WHERE c.group_id IS NOT NULL
) AS t
LEFT JOIN public.map_timed_drop_group tdg
  ON tdg.group_id = t.group_id
 AND tdg.timed_drop_id = 2122
WHERE t.rn = 1
  -- If :drop_id is NULL, keep everyone; otherwise require membership in the drop's groups
  AND (2122 IS NULL OR tdg.timed_drop_id IS NOT NULL)
ORDER BY t.user_id;
