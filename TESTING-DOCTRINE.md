# TESTING_DOCTRINE.md
# Pump Rotation System — Testing Rules

## Status: MANDATORY. Apply to all new code and retrofitted to existing modules per schedule below.

---

## Why This Document Exists

On 2026-03-20, the system classified XLE (rank #1, +39% 60d RS, Full Confirm across all
timeframes) as **Exhaustion**. One day later XLE gained +3.18% vs SPY. The call was the
system's worst misclassification to date.

The codebase had 759 tests. All passed. The bug survived multiple code review iterations.

**Root cause:** The tests verified that "input designed for State X produces State X."
No test verified that "input qualifying for BOTH State X AND State Y resolves to the
correct winner." The state classifier is a 22-path priority cascade. The tests treated
it as 7 independent categories. The bug lived in the boundary between Step 4 (Exhaustion)
and Step 6b (Sustained Leader), a boundary that no test covered because the test spec
was designed against the conceptual model (6 states), not the actual implementation
(22 return paths with ordering dependencies).

This doctrine exists to prevent this class of failure permanently.

---

## Rule 1: Three Test Types for Every Cascade Function

A "cascade function" is any function where:
- Multiple conditions are checked in sequence
- First match wins (if/elif chain, or early return)
- The ordering between checks determines the output

**Current cascade functions in this codebase:**

| Function | File | Return Paths | Current Tests | Collision Tests |
|----------|------|-------------|---------------|-----------------|
| `_determine_state()` | `engine/state_classifier.py` | 22 | 33 | 29 |
| `map_trade_state()` | `engine/trade_state_mapper.py` | 19 | 24 | 0 |
| `assess_exit()` | `engine/exit_monitor.py` | 8 | 34 | 0 |
| `classify_regime()` | `engine/regime_gate.py` | 3 | 57 | n/a (3 paths) |
| `classify_catalyst()` | `engine/catalyst_gate.py` | 5 | 13 | 0 |

Every cascade function requires three types of tests:

### Type A: Happy Path (existing pattern — keep these)
One test per output state/action. Input is designed to match exactly one branch.
```
"Input for Broadening -> Broadening"
```

### Type B: Collision Tests (THE MISSING TYPE)
One test per pair of adjacent branches in the cascade. Input must satisfy
the criteria for BOTH branches. The test asserts which one wins.
```
"Input qualifying for both Exhaustion AND Broadening -> Broadening wins
 because sustained leader exemption has higher priority"
```

How to generate these mechanically:
1. List every `return` statement in the function, in order
2. For each consecutive pair (return N, return N+1), ask:
   "Can any input reach both conditions simultaneously?"
3. If yes, write a test with that input, asserting the correct winner
4. If no (conditions are mutually exclusive), document why

### Type C: Boundary Tests
One test per numeric threshold in each branch. Input is at the exact boundary value.
```
"delta = 0.005 (exactly _DELTA_NEAR_ZERO) -> verify which side it lands on"
"rev_pctl = 75.0 (exactly above_75th boundary) -> verify consistent treatment"
"pump_percentile = 75.0 (exactly min_overt_pctl) -> verify Overt Pump or not"
```

### Naming Convention
```
tests/unit/test_{module}_collisions.py    -- Type B tests
tests/unit/test_{module}_boundaries.py    -- Type C tests
tests/unit/test_{module}_synthetic.py     -- Type A tests (existing)
```

---

## Rule 2: Parameter Coverage Audit

Every function with >5 parameters must have a coverage audit showing which
parameters are actually varied across the test suite. If a parameter appears
at its default value in >80% of tests, it is undertested.

### Coverage targets

| Parameter type | Minimum coverage |
|---------------|-----------------|
| Parameters that existed in Phase 1 (pump, rank, percentile, delta_history) | 80%+ |
| Parameters added in Phase 2+ (reversal, concentration, rs_5d/20d/60d, horizon) | 40%+ |
| Parameters that change cascade routing (regime, reversal above_75th, horizon) | 60%+ |

Any parameter below its minimum triggers: write tests that exercise it.

---

## Rule 3: Spec Staleness Check

### When a module's function signature grows, the test spec is stale.

**Rule:** After adding a parameter that changes classification routing (not just
confidence), you must:

1. Run the parameter coverage audit
2. Add collision tests involving the new parameter
3. Add boundary tests for the new parameter's thresholds
4. Add at least one "full-dimensional archetype" test that sets ALL parameters
   to realistic values (not defaults)

### Full-Dimensional Archetype Tests

These tests set every parameter to a realistic value from a real or plausible
market scenario. They catch interaction effects that partial tests miss.

Every cascade function must have at least 5 archetype tests with all parameters set.
When a live misclassification occurs, add its exact values as a new archetype.

---

## Rule 4: Post-Misclassification Protocol

When the system makes a wrong call that is identified in live or paper operation:

### Step 1: Trace the full decision path (BEFORE writing any fix)
### Step 2: Write the collision test (BEFORE writing any fix) -- test MUST FAIL
### Step 3: Generalize to the collision class (not just one data point)
### Step 4: Check adjacent collisions
### Step 5: THEN fix the code and confirm all tests pass

---

## Rule 5: Cascade Modification Protocol

When modifying a cascade function (adding a step, changing ordering, adjusting thresholds):

### Before the change:
1. List every return path
2. Identify which adjacent pairs are affected
3. Verify collision tests exist for those pairs

### After the change:
1. Re-list every return path
2. Write collision tests for any NEW adjacent pairs
3. Run parameter coverage audit
4. Run ALL existing tests (no regressions)

---

## Summary

The test suite must answer three questions, not one:

| Question | Test Type | Status |
|----------|-----------|--------|
| Does input for State X produce State X? | Happy path | Covered |
| When input qualifies for both X and Y, which wins? | **Collision** | In progress |
| At exact threshold boundaries, is behavior consistent? | **Boundary** | In progress |

Fix the zeros. That's the doctrine.
