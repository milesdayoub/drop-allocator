# Discussing allocation and sponsorship ratios
_Exported on 8/12/2025 at 15:00:22 EDT from Cursor (1.4.2)_

---

**User**

The allocator doesn't seem to do anything with sponsorship ratios.
@linear.md shows some of the early project thoughts on it
@current.md is what another cursor instance said when I asked how the current allocation works (without any optimizer). Ignore anything related to "legendary." 

So our current allocation happens like this
Sponsored: one per-user coin flip → all slots match that result.
Selection: per-user, score-weighted random; no cross-user optimization.
Supply: legendary=2 is enforced; otherwise limits only if > 0. With limit=0, there's effectively no supply cap.

The big difference is the architecture. We currently start the drop and run assignments as we decide what to give our users. The goal here is to generate the list of assignments in a csv, and then take that and then proceed with the actual assigning. So yeah, don't get bogged down in the architecture of it. Here we just want input data and output data.

We want feature-parity with the sponsorship ratio aspect. What are your thoughts on how we should proceed?

---

**Cursor**

Looking at the current allocation system and the POC allocator, I can see the key gap: **sponsorship ratio enforcement**. 

The current system does a per-user coin flip based on `group.drop_sponsorship_ratio`, ensuring all slots for a user are either sponsored or non-sponsored. The POC allocator doesn't have this constraint, which means it could assign mixed sponsorship types to a single user.

Here's how I'd approach adding sponsorship ratio parity to the POC:

## Option 1: Pre-assign sponsorship slots (closest to current behavior)

1. **Pre-process the eligibility data**: For each user, randomly determine their sponsorship designation using their group's ratio
2. **Split eligibility into two datasets**: sponsored-only and non-sponsored-only per user
3. **Run the optimizer twice**: Once for sponsored assignments, once for non-sponsored
4. **Merge results**: Combine the two assignment sets

This mirrors exactly how the current system works - one decision per user, all slots match.

## Option 2: Add sponsorship uniformity constraints to the ILP

1. **Add binary variables**: `y_u` for each user indicating if they get sponsored slots
2. **Add constraints**: For each user, all assignments must respect their sponsorship designation
3. **Add ratio constraint**: Sum of sponsored users should approximately match the group's ratio

This gives the optimizer more flexibility while maintaining the business rule.

## Option 3: Post-process optimization results

1. **Run the optimizer normally** (ignoring sponsorship)
2. **Post-process assignments**: For each user, if they have mixed sponsorship types, randomly choose one type and reassign the other slots
3. **Re-optimize**: Run again with the remaining supply to fill reassigned slots

## My recommendation: Option 1

It's the most straightforward and maintains exact feature parity. The current system's "all-or-nothing" sponsorship per user is a key business rule that should be preserved.

The implementation would look like:
1. Read group sponsorship ratios
2. For each user, flip a coin based on their group's ratio
3. Create two separate eligibility matrices (sponsored vs non-sponsored)
4. Run the optimizer on each matrix separately
5. Combine results into final assignment CSV

This approach also makes the optimization problem simpler since each run only deals with one sponsorship type, potentially improving solver performance.

What do you think? Does this align with how you want to handle the sponsorship ratio requirement?

---

**User**

I want to do things the "right" way here. Here are some things that were written in the project planning doc
- City sponsorship ratio: Set growth cities to lower; set mature cities to higher.
- To have control over sponsorship ratios by market
- To add growth brands without influencing sponsorship ratio
- Per-city sponsorship ratios: Lower to increase user growth. Increase to grow revenue.

So if these are goals, what do you suggest we do to achieve them?


---

**Cursor**

Looking at those business goals, I think **Option 2 (adding sponsorship uniformity constraints to the ILP)** is the "right" way to go. Here's why:

## The Business Goals Require Flexibility

The current system's "one coin flip per user" approach is too rigid for these strategic objectives:

- **Growth cities need lower ratios** → More users get non-sponsored rewards to drive engagement
- **Mature cities need higher ratios** → More users get sponsored rewards to maximize revenue
- **Add growth brands without influencing ratios** → Need precise control over the mix

## Why Option 2 is Better

1. **Market-level control**: The optimizer can respect exact sponsorship ratios per city/group while still finding globally optimal assignments

2. **Strategic flexibility**: You can set different ratios for different markets and the optimizer will respect them deterministically (no randomness)

3. **Growth brand integration**: New brands can be added with specific sponsorship designations without disrupting existing ratio controls

4. **Revenue optimization**: The optimizer can maximize total value while respecting your business constraints

## Implementation Approach

```python
# Add to the ILP model:
for user in users:
    group = user_group_mapping[user]
    target_sponsored_ratio = group_sponsorship_ratios[group]
    
    # Binary variable: 1 if user gets sponsored slots, 0 otherwise
    y_u = model.NewBoolVar(f"sponsored_user_{user}")
    
    # Constraint: all assignments for this user must match their sponsorship type
    for contract in eligible_contracts[user]:
        if contract.is_sponsored:
            # If contract is sponsored, user must be in sponsored group
            model.Add(x[user, contract] <= y_u)
        else:
            # If contract is non-sponsored, user must NOT be in sponsored group  
            model.Add(x[user, contract] <= 1 - y_u)
    
    # Constraint: respect group sponsorship ratio
    model.Add(
        sum(y_u for u in users if user_group_mapping[u] == group) <= 
        len([u for u in users if user_group_mapping[u] == group]) * target_sponsored_ratio
    )
```

## Benefits Over Current System

- **Deterministic**: No randomness in ratio adherence
- **Precise**: Exact control over market-level sponsorship ratios
- **Scalable**: Easy to adjust ratios for new markets or changing business needs
- **Optimized**: Still gets the benefits of global optimization while respecting business rules

This approach gives you the strategic control mentioned in the planning doc while maintaining the optimization benefits. You can set growth cities to 30% sponsored, mature cities to 80% sponsored, and the optimizer will respect those exact ratios while finding the best possible assignments.

Does this align with your vision for achieving those business goals?

```python
# Add to the ILP model:
for user in users:
    group = user_group_mapping[user]
    target_sponsored_ratio = group_sponsorship_ratios[group]
    
    # Binary variable: 1 if user gets sponsored slots, 0 otherwise
    y_u = model.NewBoolVar(f"sponsored_user_{user}")
    
    # Constraint: all assignments for this user must match their sponsorship type
    for contract in eligible_contracts[user]:
        if contract.is_sponsored:
            # If contract is sponsored, user must be in sponsored group
            model.Add(x[user, contract] <= y_u)
        else:
            # If contract is non-sponsored, user must NOT be in sponsored group  
            model.Add(x[user, contract] <= 1 - y_u)
    
    # Constraint: respect group sponsorship ratio
    model.Add(
        sum(y_u for u in users if user_group_mapping[u] == group) <= 
        len([u for u in users if user_group_mapping[u] == group]) * target_sponsored_ratio
    )
```

---

**User**

great. I like that implmentation approach.
Right now the optimizer just takes in 2 input files. I've been using clean_contract_caps.csv and elig_pairs_2121_top30_strat_v2.csv.

The idea is that one file is the contract caps, the other is eligibility pairs. Your can look at the headers to see what kind of data is present. Based on the above, we probably need additional data input, right? Can you tell me what you think?

---

**Cursor**

Let me examine the current input files to understand what data we have and what we need to add for the sponsorship ratio constraints.


Now let me look at the current allocator to see what input files it expects and their structure:


Let me see more of the main function to understand the current input structure:


Now let me check what the current input files look like by examining the data directory:


Let me examine the contract caps file to see its structure:


Now let me check the eligibility pairs file structure:


Let me check if there are any smaller sample files:


Let me check the SQL file to understand the structure better:


Based on my analysis, here's what I think we need to add for sponsorship ratio constraints:

## Current Input Structure

**Contract Caps (`clean_contract_caps.csv`):**
- `contract_address`, `brand_id`, `is_sponsored`, `cap_face`, etc.

**Eligibility Pairs (`elig_pairs_*.csv`):**
- `user_id`, `contract_address`, `score`

## What We Need to Add

For the sponsorship ratio constraints, we need **one additional input file**:

### 3. Group Sponsorship Ratios (`group_sponsorship_ratios.csv`)
```
group_id,drop_sponsorship_ratio
"harvard",0.7
"mit",0.6
"nyu",0.8
"ucla",0.5
```

### 4. User-Group Mapping (`user_group_mapping.csv`) 
```
user_id,group_id
"user_123","harvard"
"user_456","mit"
"user_789","nyu"
```

## Why These Additional Files?

1. **Group Sponsorship Ratios**: This gives you the strategic control mentioned in the planning doc - set growth cities lower (0.3-0.5), mature cities higher (0.7-0.9)

2. **User-Group Mapping**: This connects users to their groups so the optimizer can enforce the sponsorship ratio constraints per group

3. **Separation of Concerns**: Keeps the business logic (sponsorship ratios) separate from the optimization data (eligibility, caps)

## Alternative: Embed in Existing Files

If you prefer to avoid additional files, we could:

1. **Add `group_id` to eligibility pairs** - but this duplicates data across many rows
2. **Add `drop_sponsorship_ratio` to eligibility pairs** - but this also duplicates data

## My Recommendation

Go with the **3 separate input files** approach:
- `caps.csv` - contract supply constraints  
- `elig.csv` - user-contract eligibility + scores
- `group_ratios.csv` - business rules for sponsorship ratios
- `user_groups.csv` - user-to-group mapping

This keeps the data clean, avoids duplication, and makes it easy to adjust sponsorship ratios for different markets without regenerating the eligibility data.

What do you think? Does this approach make sense for your use case?

```plaintext
group_id,drop_sponsorship_ratio
"harvard",0.7
"mit",0.6
"nyu",0.8
"ucla",0.5
```

```plaintext
user_id,group_id
"user_123","harvard"
"user_456","mit"
"user_789","nyu"
```

