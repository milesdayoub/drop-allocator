SELECT g.id         AS group_id,
       g.drop_sponsorship_ratio::numeric AS sponsorship_ratio
FROM   map_timed_drop_group m
JOIN   "group" g ON g.id = m.group_id
WHERE  m.timed_drop_id = 2122
  AND  g.is_active = TRUE
ORDER  BY g.id;
