Business-Side Leader

Who is the main business contact? This person should be meeting with the engineering lead twice per week (Tu/Th).

@luke

Requirements

User Messaging

How would you message this feature to the intended user?

As an internal Claim employee, I want:

Targeting

Not to have to use boosts

To maximize brand value (i.e. most conversions for globally best customers)

Guardrails

To have control over sponsorship ratios by market

To avoid going over supply limits

To avoid assigning consumers brands they clearly dislike

Infra

To add growth brands without influencing sponsorship ratio

To stop worrying if the Drop ran

To be able to quickly test these in a non-production environment

Levers: What can marketplace change to affect the Drop?

Contract scoring algorithm: How do we decide best matches?

City sponsorship ratio: Set growth cities to lower; set mature cities to higher.

Minimum affinity cutoff: Set a cutoff point where users don't get any Drop (rather than XP again)

Unsponsored contract supply: For growth cities, get better unsponsored brands on the platform

Sponsored contract pricing (i.e. how much gets passed to the consumer): To support growth, take a lower margin. To support business, take a higher margin.

As a Claim brand, I want:

The best possible customers you can give me

As a Claim user, I want:

To get good rewards: relevant brand, near me, and a good enough deal

Intention

What is the specific, time-bound impact this project is driving?

As we start to scale, we need more control over the Drop. This project is urgent because Claim is growing faster, and inefficiencies in the Drop are increasingly costly in terms of money, effectiveness for brands, and consumer experience.

By the end of this project, there should be only three factors that impact the Drop:

Per-city sponsorship ratios: Lower to increase user growth. Increase to grow revenue.

Sponsored rewards: Improve supply to increase revenue.

Growth rewards: Improve supply to increase growth and retention.

Deliverables

What tangible product changes will this project make? How will we know it's done?

This project will be done when:

New assignment logic runs with globally optimized brand

Guardrails are respected (eligibility, city mapping, supply limits)

Recommender is confirmed working E2E

Tests are added

The dashboard is updated

Monitoring is added

Out of Scope

What potential product changes will this project not make? Why not? Are there conditions where something might come in-scope?

This project will not add A/B testing or holdouts

Communications

Please list kickoff and closing meeting dates, and link Slack channel (format: #project-xxx) for this project below.

#project-update-drop-logic

Planning

Key Features

What are the features needed for the project? What are their relative priorities? What specific actions or abilities will the features enable?

New scoring: We'll simplify scoring to be based on five factors. All factors are kept in the 0.01-1 range with mathematical weights applied to give each factor its proper influence while preventing any one factor from completely dominating the final score.

Brand affinity (8x): The most important factor. These scores will be from the recommender (which retrains weekly).

Distance (2x): We'll compare a brand's distance to the median distance of available brands; this ensures our system works in both New York (dense) and Dallas (dispersed). Note: This requires us to assign by cities.

Reward value (1x): Higher rewards have higher value (likely tanh function).

Rejection history (1x): Brands that the user has previously rejected receive a lower score. Ex: Each rejection decreases likelihood by ~20%.

Weeks since last app open (1x): The longer a user hasn't opened the app, the lower their score. Ex: Reduce score by 20% for each week since last app open. Note: This needs to be app open (vs. mint Drop) to avoid self-reenforcing loop.

Brand-first assignment: We want to ensure that brands get the best users (and that the best users get their favorite brands).

Per-city sponsorship ratios: Cities will have individual sponsorship ratios (see brand-first assignment below).

Brand ranking minimums: If a user's assignments are all within the bottom X% of their rankings, skip them in the Drop.

Supply limit calculations: TODO. Need to account for funnels (selection + conversions). Also need to take into account existing tokens and conversions.

Could be hard-coded (or sensible defaults), but ideally we automatically update as we learn more

We probably target [80%] fulfillment of supply to avoid going over

Automated rollover: At the end of a month, if a contract is meant to be monthly, we automatically clone it over.

Test environment: We need to be able to run a test Drop in a non-production environment and fast enough to run multiple iterations.

Engineering Plan

What specific actions will engineering need to take? What will they need to build?

Backend

Skip assignment if there are no brands outside the bottom X% of user preference (X% by city)

Add (or confirm) max supply limits

Add max conversion limits (ex: if Shake Shack has completed their compaign, stop assigning)

Add (or confirm) city mapping

Add (or confirm) eligibility checks

Tech debt

Remove boosts from scoring

Remove excluded contracts from filtering

Remove location Drop assignment logic

Tests

Add test seed functions that simulate the dashboard

Adding a Drop

Adding a contract

Updating a Drop

Setting sponsorship ratio

Add integration test: Creating a Drop

Add integration test: Running a Drop

Add unit test: Drops respect sponsorship ratio per city

Add unit test: Drops do not assign ineligible rewards

Add unit test: Drops do not assign is_onboarding rewards

Add unit test: Drops only assign is_drop=true rewards

Add unit test: Drops assign zero rewards if conversion maximum is hit

Dashboard

Remove Boosts from Drop edit page

Remove excluded contracts from Drop edit page

Remove ability to set type to location Drop

Add warning for changing Drops per user

Add warning for changing city sponsorship ratio

Add post-drop sponsorship ratio to a Drop that has completed (based on selections, not assignments)

[design needed] Add Drop preview page

Operations

Confirm changes with Growth team

Confirm changes with Sales team

Confirm changes with Marketplace team

Monitoring

Add New Relic alert: Drop interrupted

Add New Relic alert: Recommender did not retrain

Add New Relic alert: Received zero assignments in the Drop

Add Slack post to #droperations when Drop completes with

Assignments created

Pct users without assignments

Sponsorship ratio

Add Slack post when recommender retrains

Ops Plan

What actions will the rest of the team (ex: BD, CS, growth) need to take?

Success Spec

How will we understand if the project is successful? Please use specific, falsifiable metrics.

We'll know the project is successful if:

Growth only adds and adjusts growth contracts

Sales only adds and adjusts sales contracts

Marketplace only adjusts sponsorship ratios and other levers

No one worries about the Drop running

The team has clarity on how the Drop works

Deadline

When do we need this project completed? Why are we choosing that deadline?

Timeline Risks

Are there any known potential issues that might delay the project? How could we mitigate?

Internal Stakeholders

Who absolutely needs to sign off on this plan before building starts? Make this list as short as possible.

@tap

@luke

@dyan

@john

Deployment

Launch Checklist

What are our final to-dos and checks before deploying to prod?

Tests passing

Run test drop in staging

Review setup with

Sales

Growth

Marketplace

Review results with

Sales

Growth

Marketplace

Merge to main

Deploy

Post-Launch Checklist

How will we know each feature is working after it is deployed?

Confirm New Relic logs running

Confirm Slack messages running

Post-Launch Messaging

How should we announce the change? Are there any communication or marketing needs? When do we need to notify marketing by?

N/A

Graveyard

For every city, sorted by size

For every contract and eligible-user pairing, run scoring (see new scoring above)

For every user, get rankings of eligible contracts

For every contract, get rankings of eligible users

Cut out each users's bottom X% of rankings (see brand-affinity minimums above)

Run the residency assignment algorithm

Users "apply" to their #1 contract

Contract takes their favorite users (up to the contract cap) that have ranked the contract #1

The users are removed from consideration for other contracts from that brand

The round repeats for users' #2 contract

Once all sponsored supply is allocated, calculate the number of unsponsored assignments we can give out while maintaining the city's sponsorship ratio.

To allocate the unsponsored assignments

Find users with 2 open slots. Sort by newest users first (if possible: with some randomness). Assign them a (weight random selected) top unsponsored reward

If unsponsored supply remaining, find users with 1 open slot. Sort by newest users first (if possible: with some randomness). Assign them a (weight random selected) top unsponsored reward.

Note: This implies that you can only get the Drop if you have min. 1 sponsored reward

Implementation Details

ML Scoring

Is a User's Brand Affinity the same thing as a Brand's Best Customer?

User's Brand Affinity: Highest probability of user conversion

Brand's Best Customer: Highest probability that the user will come back without reward (LTV/ROI)

Decision: Are we going to somehow train two different models?

One model?

Solving this 80% seems to be super important

Could be reasonable to believe 1 week of two FTE time

Data Setup, model design, training, pipeline setup, etc.

Assignment Selection Algorithm

Does the way in which Brands "rank" differ by brand?

ChatGPT/Google Research: Given some sort of score, is the Residency Matching algorithm the right algorithm to use with derived rankings/scores vs stated preferences? Is there a simpler matching algo that can get the job done?

Pros/Cons

Residency Matching Algorithm

How much to weigh Brands vs. User?

Forces 2 different model scores

More steps from score to final scores (vs. just seeing a score list)

Do we really know the preference of user's and brand's?

Hypothesis: Not as good at "maximizing utility"

"Single" ML generated score

"utility monster" problem

Confounding factors into one score

Harder to tune (more guessing to get our desired result)

Infrastructure
