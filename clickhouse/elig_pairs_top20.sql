-- 2121: 2025-08-14 15:00:00 → 2025-08-15 07:00:00 UTC
-- Top 20 eligible contracts per user (by recommender score)
SELECT DISTINCT
       t.user_id,
       t.contract_address,
       t.score
FROM
(
    SELECT
        uce.user_id                         AS user_id,
        uce.contract_address                AS contract_address,
        COALESCE(ubr.score, 0.0001)        AS score,
        row_number() OVER
        (
            PARTITION BY uce.user_id
            ORDER BY COALESCE(ubr.score, 0.0001) DESC, uce.contract_address
        ) AS rn
    FROM public_user_contract_eligibility AS uce
    INNER JOIN public_contract AS c
            ON c.address = uce.contract_address
    LEFT  JOIN public_user_brand_recommendation AS ubr
            ON ubr.user_id = uce.user_id
           AND ubr.brand_id = c.brand_id
    WHERE uce.is_eligible = 1
      AND c.is_drop       = 1
      AND c.start_datetime <= toDateTime('2025-08-15 07:00:00','UTC')
      AND c.end_datetime   >= toDateTime('2025-08-14 15:00:00','UTC')
) AS t
WHERE t.rn <= 20
FORMAT CSVWithNames
SETTINGS max_result_rows = 0, max_result_bytes = 0, max_execution_time = 0;
