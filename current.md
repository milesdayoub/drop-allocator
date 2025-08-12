# Reviewing drop assignment optimization strategies

_Exported on 8/12/2025 at 11:16:59 EDT from Cursor (1.4.2)_

---

**User**

We're building a new way to decide what users get in the drop. Could you go through this POC for using an optimizer/solver and compare it to how the Drop is currently being done in claim/core?

Here is an output that I had from a past chat about how it works today
High-Level Drop Assignment Allocation Overview
Core Process Flow
The Drop assignment system follows a sophisticated multi-stage process to fairly distribute contracts to users:

1. Drop Initialization & Group Processing
   Timed Drops are triggered by scheduled jobs that identify queued drops
   Groups are processed in priority order - largest groups (by user count) get processed first to ensure fair access to limited supply contracts
   Only active groups participate in drops, regardless of configuration
2. User Batching Strategy
   Users are processed in batches of 100 to prevent system overload
   Within each batch, users are processed in chunks of 1,000 for memory management
   Random group order ensures no geographic bias in contract allocation
3. Contract Eligibility & Filtering
   Contract-Level Eligibility:
   Must not be REWARD type contracts (system contracts)
   Cannot be in the drop's excluded contracts list
   Must respect supply limits (administrator-configured or automatic legendary limits)
   Must match brand-group geographic mapping
   User-Level Eligibility:
   Spending cap rules: Users with $50+ payouts can only get sponsored/common contracts (unless 10+ invites)
   Recent brand prevention: No contracts from brands assigned/claimed/rejected in last 8 days
   Token ownership blocking: No contracts where user has unexpired tokens
   Account status: Must not be suspended
   Group membership: User's primary group must overlap with brand availability
4. Assignment Slot Determination
   This is a key innovation in the system:
   Sponsorship ratio application: Each group has a drop_sponsorship_ratio (e.g., 70% for Harvard)
   Random determination: For each user, the system randomly determines if they'll receive sponsored or non-sponsored contracts based on their group's ratio
   Slot creation: Creates assignment slots (typically 3) with the same sponsored/non-sponsored designation for that user
   Example:
   scoring
5. Scoring Algorithm
   The system uses a sophisticated weighted scoring system:
   Base Score Factors:
   Previous brand interactions: Decrements for each time user has engaged with the brand
   Assignment history: Decrements for previous assignments to same contract
   Rarity scaling: Non-sponsored contracts scaled by rarity (Common=1, Legendary=0.01)
   ML recommendation score: AWS-powered recommendation system with exponential weighting
   New user boost: 1000x boost for food rewards if user is within first 15 days
   Score Calculation:
   new_user_boost
6. Contract Selection Process
   For Each User:
   Filter assignable contracts that haven't reached supply limits
   Apply slot constraints (sponsored vs non-sponsored)
   Ensure brand diversity - no multiple contracts from same brand
   Weighted random selection based on calculated scores
   Track supply usage in real-time to prevent over-allocation
7. Supply Limit Management
   Real-time enforcement:
   Dynamic tracking: Counts existing + pending assignments during processing
   Automatic legendary limits: Legendary contracts limited to 2 assignments per drop
   Administrator limits: CS Dashboard configured limits (0 = unlimited)
   Fair distribution: Random group processing prevents any group from consistently getting first access
8. Assignment Creation
   Final assignments are created with:
   Contract details: Address, token value, attributes
   User association: Linked to specific user and timed drop
   Expiration handling: Default 7-day expiry for tokens
   Reward source tracking: Marked as DROP source for analytics
   Key Fairness Mechanisms
   Geographic Fairness: Random group processing order
   Sponsored Balance: Group-level sponsorship ratios ensure appropriate mix
   Brand Diversity: No user gets multiple contracts from same brand
   Supply Fairness: Real-time tracking prevents over-allocation
   User Preference: ML-powered recommendation scoring
   New User Support: Special boosts for early-stage users
   System Architecture Notes
   Two assignment systems exist: The newer sophisticated system (Assignment/assignment.ts) and a simpler legacy system (DropCheckIn/assignment.ts) marked as technical debt
   Batch processing: Handles thousands of users efficiently
   Event-driven: Integrates with Kafka for analytics and monitoring
   Database optimization: Uses read replicas and efficient querying patterns
   This system ensures fair, personalized, and scalable distribution of rewards to users while maintaining business rules around sponsorship, supply limits, and user engagement patterns.

---

**Cursor**

I’m going to scan the core service to locate the current drop assignment implementation and related business rules (eligibility, sponsorship ratios, scoring, supply limits). I’ll search broadly, then narrow down to the key files for an accurate comparison to the POC optimizer.

I'll load the core assignment and timed drop files to map the current algorithm (batching, sponsorship slots, scoring, supply limits) and compare them to the POC optimizer.

I reviewed the current Drop assignment flow in `claim/core` and compared it to the POC optimizer in `poc/src/allocator/main.py`.

### What core does today (key mechanics)

- **Batching and group order**
  - Processes active groups, sorted by descending user count (not random).

```118:152:claim/core/src/services/TimedDrop/timedDrop.ts
// Process groups in descending order by user count (most users first).
...
groupIds = orderBy(
    groupUserCounts,
    (g) => g._count.map_user_group,
    'desc'
).map((g) => g.id);
```

- **Eligible contract universe per group**
  - Group-to-brand mapping, current time window, `is_drop`, excludes TD-specific excluded list.

```16:43:claim/core/src/services/TimedDrop/possibleContracts.ts
const groupContracts = await prismaRO.contract.findMany({
    where: {
        brand: { map_brand_group: { some: { group: { id: group_id } } } },
        start_datetime: { lte: now },
        end_datetime: { gt: now },
        is_drop: true,
    },
    ...timedDropContractInclude,
});
...
possibleContracts = groupContracts.filter(
    (c) => c?.address && !excludedContractAddresses.includes(c.address)
);
```

- **Eligibility and scoring**
  - Uses precomputed `user_contract_eligibility` for user-level rules (token ownership, recent brand prevention, account status, etc.), then scores.
  - Scoring: history penalties, recommender score with exponent, rarity scaling for non-sponsored, “new user food” boost.

```148:169:claim/core/src/services/Assignment/assignment.ts
this.scoredEligibilities.push({
  ...agg,
  contract,
  score: this.calculateScore(user, contract),
  experiment_id: experimentId || undefined,
});
```

```267:327:claim/core/src/services/Assignment/assignment.ts
// penalties for past brand/assignments → convert to 0-1
// multiply by recommender score ^ exponent
// 100x boost for restaurants if user < 15 days
// scale by rarity if non-sponsored
```

- **Sponsored vs non-sponsored slots per user**
  - Draw once per user with `group.drop_sponsorship_ratio`; all slots for that user share the same designation.

```497:511:claim/core/src/services/Assignment/assignment.ts
const isSponsored = Math.random() <= sponsorshipRatio;
const slots: AssignmentSlot[] = Array.from(
  { length: this.assignmentsPerUser },
  () => ({ isSponsoredContract: isSponsored })
);
```

```201:205:claim/core/src/services/TimedDrop/timedDrop.ts
let sponsorshipRatio: number = Number(groupDetails.drop_sponsorship_ratio);
```

- **Selection policy**
  - For each user: filter to slot designation, enforce brand diversity (no duplicate brands), do weighted-random pick by score.

```524:560:claim/core/src/services/Assignment/assignment.ts
filteredContracts = filteredContracts.filter(
  (a) => a.contract.is_sponsored === slot.isSponsoredContract
);
filteredContracts = filteredContracts.filter(
  (a) => a.contract.brand_id && !usedBrandIds.has(a.contract.brand_id)
);
const selectedContracts = this.weightedRandomSelect(filteredContracts, 1);
```

```187:224:claim/core/src/services/Assignment/assignment.ts
// weightedRandomSelect by score, also prunes same-brand for subsequent picks
```

- **Supply constraints**
  - Tracks in-DB current supply for the Drop, enforces per-contract limits and a special legendary limit of 2.

```404:416:claim/core/src/services/Assignment/assignment.ts
const aggregate = await prisma.contract_assignment.groupBy({
  by: ['contract_address'], _count: true, where: { timed_drop_id: this.timedDrop.id },
});
this.currentSupply = aggregate.reduce(...);
```

```477:495:claim/core/src/services/Assignment/assignment.ts
if (contract.rarity === LEGENDARY) limit = 2; else limit = configured per-contract
return limit > 0 && currentSupply >= limit;
```

- **Assignment creation**
  - Writes to DB with `timed_drop_id`, assignment window aligned to Drop.

```451:464:claim/core/src/services/Assignment/assignment.ts
assignmentInput.timed_drop_id = this.timedDrop.id;
assignmentInput.expiration_datetime = this.timedDrop.end_time;
assignmentInput.assigned_datetime = assignedDatetime;
```

- **Batching**
  - Processes users in batches of 100 inside 1000-sized chunks.

```38:51:claim/core/src/services/Assignment/batchedTimedDropAssignment.ts
for (let i = 0; i < userIds.length; i += 1000) { const subset = userIds.slice(i, i + 1000);
  for (let i = 0; i < subset.length; i += this.BATCH_SIZE) { /* BATCH_SIZE = 100 */ ... }
}
```

### What the POC optimizer does

- Reads CSVs (`caps`, `elig`) and maximizes score with constraints:
  - Per-user ≤ k; per-contract ≤ capacity.
  - Adds optional coverage encouragement:
    - z-coverage variables per user and slot with weights; or
    - shortfall penalty mode to push everyone toward k.
- Supports warm start (greedy), pre-trimming per-user top-N, and min-score filtering.
- Can fall back to greedy if solver fails.
- Output is a CSV; no DB integration.

### Gaps vs core business rules

- Sponsored slot policy:
  - Core: per-user single sponsored/non-sponsored decision, all slots match. POC: no per-user sponsored uniformity; it may mix sponsored and non-sponsored for a single user.
- Brand diversity:
  - Core enforces one brand per user. POC has no per-user, per-brand ≤ 1 constraint.
- Legendary special rule:
  - Core caps LEGENDARY to 2 per Drop. POC has generic per-contract caps but no special legendary handling.
- Eligibility depth:
  - Core eligibility includes multiple business checks (recent brand, token ownership, account status, geography, spending caps) via `user_contract_eligibility`. POC assumes elig CSV already encodes all such rules.
- Scoring parity:
  - Core uses history penalties, recommender exponent, rarity scaling, and new-user food boost (100x in code). POC treats input `score` as-is.
- Group processing and fairness:
  - Core processes groups largest-first. POC has no grouping; fairness is encouraged via coverage weights/shortfall penalty instead.
- Supply configuration:
  - Core uses per-contract limits from the DB and 0=unlimited; POC synthesizes capacity for unsponsored rows (`--unsponsored_cap`), which differs from DB-driven limits.

### How to align the POC to core (minimal changes)

- Sponsored uniformity per user:
  - Add a binary `y_u` per user; for each elig row i with sponsored flag s_i:
    - x_i ≤ y_u if s_i=1; x_i ≤ (1 - y_u) if s_i=0.
  - To match today’s behavior (Bernoulli per user) either:
    - Pre-sample `y_u` from the group ratio and fix it, or
    - Constrain sum_sponsored / total ≈ ratio at group level if you want deterministic adherence.
- Brand diversity:
  - For each user u and brand b: sum over contracts of brand b for user u ≤ 1.
- Legendary limit:
  - Set capacity for legendary contracts to 2 for the Drop.
- Eligibility:
  - Build `elig.csv` from `user_contract_eligibility` and `possibleContracts()` to reflect all existing business rules.
- Scoring:
  - Reproduce `calculateScore()` in the data pipeline to compute the `score` column identically to core.
- Supply limits:
  - Use `timed_drop_contracts.limit` and current DB counts for `cap_face`. For unlimited (0), either set a very large cap or drop the constraint.
- Batching:
  - Run per group to mirror current sequencing; set k = `timed_drop.contracts_per_user`.

### Notable doc vs code mismatches

- New user food boost: Docs mention 1000x; code applies 100x.
- Group ordering: Docs say “random group order”; code uses descending by group size.

### Where this lands strategically

- POC gives global optimality and explicit fairness knobs (coverage/shortfall), but needs the above constraints to match product rules.
- Core is fast, randomized, and DB-native, but not globally optimal.

- I can wire a thin adapter to pull core data into the POC (eligibility, caps, scoring), add the missing constraints, and write results back as assignments. If you want, I’ll sketch the exact SQL/Prisma queries and a batch runner.

- Status: Mapped core flow and rules to code; identified deltas with the POC; outlined concrete constraint changes to make the POC equivalent while preserving solver advantages.

- Key points:
  - Core: weighted-random with brand diversity, per-user sponsored uniformity, DB-driven limits, and eligibility; batched by group size.
  - POC: ILP/CP-SAT over CSVs with capacity and coverage; missing sponsored uniformity, brand diversity, legendary special rule, and core’s scoring/eligibility unless added.
  - Two doc mismatches with code: new user boost magnitude and group processing order.

```typescript
// Process groups in descending order by user count (most users first).
...
groupIds = orderBy(
    groupUserCounts,
    (g) => g._count.map_user_group,
    'desc'
).map((g) => g.id);
```

```typescript
const groupContracts = await prismaRO.contract.findMany({
    where: {
        brand: { map_brand_group: { some: { group: { id: group_id } } } },
        start_datetime: { lte: now },
        end_datetime: { gt: now },
        is_drop: true,
    },
    ...timedDropContractInclude,
});
...
possibleContracts = groupContracts.filter(
    (c) => c?.address && !excludedContractAddresses.includes(c.address)
);
```

```typescript
this.scoredEligibilities.push({
  ...agg,
  contract,
  score: this.calculateScore(user, contract),
  experiment_id: experimentId || undefined,
});
```

```typescript
// penalties for past brand/assignments → convert to 0-1
// multiply by recommender score ^ exponent
// 100x boost for restaurants if user < 15 days
// scale by rarity if non-sponsored
```

```typescript
const isSponsored = Math.random() <= sponsorshipRatio;
const slots: AssignmentSlot[] = Array.from(
  { length: this.assignmentsPerUser },
  () => ({ isSponsoredContract: isSponsored })
);
```

```typescript
let sponsorshipRatio: number = Number(groupDetails.drop_sponsorship_ratio);
```

```typescript
filteredContracts = filteredContracts.filter(
  (a) => a.contract.is_sponsored === slot.isSponsoredContract
);
filteredContracts = filteredContracts.filter(
  (a) => a.contract.brand_id && !usedBrandIds.has(a.contract.brand_id)
);
const selectedContracts = this.weightedRandomSelect(filteredContracts, 1);
```

```typescript
// weightedRandomSelect by score, also prunes same-brand for subsequent picks
```

```typescript
const aggregate = await prisma.contract_assignment.groupBy({
  by: ['contract_address'], _count: true, where: { timed_drop_id: this.timedDrop.id },
});
this.currentSupply = aggregate.reduce(...);
```

```typescript
if (contract.rarity === LEGENDARY) limit = 2; else limit = configured per-contract
return limit > 0 && currentSupply >= limit;
```

```typescript
assignmentInput.timed_drop_id = this.timedDrop.id;
assignmentInput.expiration_datetime = this.timedDrop.end_time;
assignmentInput.assigned_datetime = assignedDatetime;
```

```typescript
for (let i = 0; i < userIds.length; i += 1000) { const subset = userIds.slice(i, i + 1000);
  for (let i = 0; i < subset.length; i += this.BATCH_SIZE) { /* BATCH_SIZE = 100 */ ... }
}
```

---

**User**

Ok thanks for providing that. I want to probe more about the current state.

You said

1. Sponsored vs non-sponsored slots per user
   Draw once per user with group.drop_sponsorship_ratio; all slots for that user share the same designation.
   Can you explain this more?

2. Selection policy
   For each user: filter to slot designation, enforce brand diversity (no duplicate brands), do weighted-random pick by score.
   Based on what you see, what's the overall layman's terms algo here? I don't see any logic that does comparisons of "who gets the best contracts" other than a score? How does this actually work?

3. Supply constraints.

I don't think we respect this in reality
Here's the output from our db about some past drops:
timed_drop_id, contract_address, id, boost, limit
2119 0x00000000000000000000000000000000000577 17197 50 0
2119 0x00000000000000000000000000000000000573 17195 50 0
2119 0x00000000000000000000000000000000000494 17188 250 0
2119 0x00000000000000000000000000000000000498 17201 50 0
2119 0x00000000000000000000000000000000000568 17194 5 0
2119 0x00000000000000000000000000000000000595 17202 500 0
2119 0x00000000000000000000000000000000000502 17189 50 0
2119 0x00000000000000000000000000000000000512 17191 50 0
2119 0x00000000000000000000000000000000000546 17193 10 0
2119 0x00000000000000000000000000000000000544 17192 10 0
2119 0x00000000000000000000000000000000000579 17198 100 0
2119 0x00000000000000000000000000000000000503 17190 50 0
2119 0x00000000000000000000000000000000000590 17200 25 0
2119 0x00000000000000000000000000000000000496 17196 50 0
2119 0x00000000000000000000000000000000000585 17199 50 0
2119 0x00000000000000000000000000000000000539 17187 100 0
2119 0x00000000000000000000000000000000000538 17186 100 0
2119 0x00000000000000000000000000000000000587 17203 50 0
2118 0x00000000000000000000000000000000000546 17028 10 0
2118 0x00000000000000000000000000000000000503 17025 50 0
2118 0x00000000000000000000000000000000000512 17026 50 0
2118 0x00000000000000000000000000000000000592 17038 50 0
2118 0x00000000000000000000000000000000000579 17033 100 0
2118 0x00000000000000000000000000000000000502 17024 50 0
2118 0x00000000000000000000000000000000000574 17031 50 0
2118 0x00000000000000000000000000000000000568 17029 5 0
2118 0x00000000000000000000000000000000000587 17036 500 0
2118 0x00000000000000000000000000000000000590 17037 25 0
2118 0x00000000000000000000000000000000000539 17023 100 0
2118 0x00000000000000000000000000000000000538 17022 100 0
2118 0x00000000000000000000000000000000000585 17034 50 0
2118 0x00000000000000000000000000000000000494 17021 250 0
2118 0x00000000000000000000000000000000000573 17030 50 0
2118 0x00000000000000000000000000000000000586 17035 500 0
2118 0x00000000000000000000000000000000000544 17027 10 0
2118 0x00000000000000000000000000000000000577 17032 50 0
2117 0x00000000000000000000000000000000000568 16928 5 0
2117 0x00000000000000000000000000000000000585 16910 100 0
2117 0x00000000000000000000000000000000000545 16911 10 0
2117 0x00000000000000000000000000000000000562 16929 50 0
2117 0x00000000000000000000000000000000000591 16922 100 0
2117 0x00000000000000000000000000000000000573 16930 100 0
2117 0x00000000000000000000000000000000000592 16923 100 0
2117 0x00000000000000000000000000000000000576 16917 10 0
2117 0x00000000000000000000000000000000000555 16913 50 0
2117 0x00000000000000000000000000000000000515 16916 100 0
Since limit is all 0, do we actually do anything with supply in reality?

---

**Cursor**

- Quick update: I dug into the exact spots in core where sponsorship slots, per-user selection, and supply limits are applied, and tied them to what you’re observing in data.

### 1) Sponsored vs non-sponsored “slots per user”

- Mechanic: one coin flip per user using the group’s `drop_sponsorship_ratio`. All of that user’s slots inherit the same result.

```497:511:claim/core/src/services/Assignment/assignment.ts
// Randomly determine if the user will receive a sponsored assignment
const isSponsored = Math.random() <= sponsorshipRatio;

// Create one slot per this.assignmentsPerUser (all the same isSponsored)
const slots: AssignmentSlot[] = Array.from(
  { length: this.assignmentsPerUser },
  () => ({ isSponsoredContract: isSponsored })
);
```

- Where the ratio comes from:

```201:217:claim/core/src/services/TimedDrop/timedDrop.ts
let sponsorshipRatio: number = Number(groupDetails.drop_sponsorship_ratio);
...
if (sponsorshipRatio > 1 || sponsorshipRatio < 0 || isNaN(sponsorshipRatio)) {
  sponsorshipRatio = 1;
}
```

- Effect in plain terms: If a group’s ratio is 0.7, about 70% of users (in that group) will get all-sponsored slots, and ~30% will get all-non-sponsored slots for that Drop. There isn’t per-slot mixing; the decision is per-user.

### 2) Selection policy (layman’s terms)

- What happens for each user:
  - Take all eligible contracts for that user (pre-filtered by business rules and time-window).
  - Apply the slot’s sponsored/non-sponsored constraint.
  - Enforce brand diversity: remove any remaining contracts from brands already picked for this user.
  - Pick 1 per slot using weighted random by the computed score (not “highest wins,” but higher score → higher chance).

```524:560:claim/core/src/services/Assignment/assignment.ts
// match sponsorship for the slot
filteredContracts = filteredContracts.filter(
  (assignable) => assignable.contract.is_sponsored === slot.isSponsoredContract
);
// enforce one brand per user
filteredContracts = filteredContracts.filter(
  (assignable) =>
    assignable.contract.brand_id &&
    !usedBrandIds.has(assignable.contract.brand_id)
);
// choose by weighted random (score-weighted)
const selectedContracts = this.weightedRandomSelect(filteredContracts, 1);
```

```187:224:claim/core/src/services/Assignment/assignment.ts
// weightedRandomSelect: draw by score, remove same-brand for subsequent picks
```

- There’s no global “best K contracts across all users.” It’s per user, score-weighted randomness. Users with higher scores for a contract are more likely to get it, but not guaranteed; and there’s no cross-user optimization.

### 3) Supply constraints in practice

- What the code enforces:
  - Legendary contracts have a hard limit of 2 per Drop.
  - Non-legendary limits are only enforced if a positive limit is configured; 0 means “ignore limit.”

```477:495:claim/core/src/services/Assignment/assignment.ts
if (contract.rarity === contract_rarity.LEGENDARY) {
  limit = 2;
} else {
  limit = this.limit?.[contract.address] || 0; // 0 = no limit
}
return (limit > 0 && currentSupply >= limit);
```

- Current supply is counted from DB for the Drop plus what’s been selected in the running batch:

```404:416:claim/core/src/services/Assignment/assignment.ts
const aggregate = await prisma.contract_assignment.groupBy({
  by: ['contract_address'],
  _count: true,
  where: { timed_drop_id: this.timedDrop.id },
});
this.currentSupply = aggregate.reduce((acc, curr) => {
  acc[curr.contract_address] = curr._count; return acc;
}, {});
```

- Your DB dump shows `limit = 0` for those contracts. That means the system treats them as unlimited, so no cap is applied (except the special legendary=2 rule). So yes: with limit=0 we do not restrict supply; the only “fairness” constraint still in effect is brand diversity at the per-user level.

- If we want true supply control, we need to start setting positive `timed_drop_contracts.limit` values (or implement a default global cap). Otherwise popular contracts can be assigned broadly.

- Summary:

  - Sponsored: one per-user coin flip → all slots match that result.
  - Selection: per-user, score-weighted random; no cross-user optimization.
  - Supply: legendary=2 is enforced; otherwise limits only if > 0. With `limit=0`, there’s effectively no supply cap.

- Done:
  - Explained per-user sponsorship slots with code refs.
  - Clarified selection mechanics in plain terms, highlighting lack of global optimization.
  - Confirmed your observation: with `limit=0`, supply isn’t capped (except legendary).

```typescript
// Randomly determine if the user will receive a sponsored assignment
const isSponsored = Math.random() <= sponsorshipRatio;

// Create one slot per this.assignmentsPerUser (all the same isSponsored)
const slots: AssignmentSlot[] = Array.from(
  { length: this.assignmentsPerUser },
  () => ({ isSponsoredContract: isSponsored })
);
```

```typescript
let sponsorshipRatio: number = Number(groupDetails.drop_sponsorship_ratio);
...
if (sponsorshipRatio > 1 || sponsorshipRatio < 0 || isNaN(sponsorshipRatio)) {
  sponsorshipRatio = 1;
}
```

```typescript
// match sponsorship for the slot
filteredContracts = filteredContracts.filter(
  (assignable) => assignable.contract.is_sponsored === slot.isSponsoredContract
);
// enforce one brand per user
filteredContracts = filteredContracts.filter(
  (assignable) =>
    assignable.contract.brand_id &&
    !usedBrandIds.has(assignable.contract.brand_id)
);
// choose by weighted random (score-weighted)
const selectedContracts = this.weightedRandomSelect(filteredContracts, 1);
```

```typescript
// weightedRandomSelect: draw by score, remove same-brand for subsequent picks
```

```typescript
if (contract.rarity === contract_rarity.LEGENDARY) {
  limit = 2;
} else {
  limit = this.limit?.[contract.address] || 0; // 0 = no limit
}
return limit > 0 && currentSupply >= limit;
```

```typescript
const aggregate = await prisma.contract_assignment.groupBy({
  by: ['contract_address'],
  _count: true,
  where: { timed_drop_id: this.timedDrop.id },
});
this.currentSupply = aggregate.reduce((acc, curr) => {
  acc[curr.contract_address] = curr._count;
  return acc;
}, {});
```
