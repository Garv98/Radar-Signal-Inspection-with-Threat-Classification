# Reward Function — Mathematical Justification

The gatekeeper reward implements **Value of Information (VoI)**: pay the cost of
a deep ML inspection only when the expected decision value of the result exceeds
that cost.

## Definitions

For a signal with true label `y` and ML posterior `p ∈ Δ^{K-1}` (K classes):

| Symbol | Definition | Meaning |
|--------|------------|---------|
| `c` | `max_k p_k` | confidence (top-1 probability) |
| `H̄` | `-Σ_k p_k ln p_k / ln K` ∈ [0,1] | normalised entropy (uncertainty) |
| `correct` | `1[argmax p = y]` | is the ML readout right |
| `certainty` | `α·c + (1-α)·(1-H̄)` | blended decisiveness |
| `v[y]` | data-derived class value | operational stakes of class `y` |
| `κ` | `compute_cost` | cost of one ML inference (utility units) |
| `β` | `miss_aversion` | false-negative asymmetry multiplier |
| `ρ, ρ*` | running / target forward rate | budget regulariser |
| `η` | `budget_penalty` | budget pressure coefficient |

### ML decision utility

```
U_ml(X) = v[y] · (2·correct − 1) · certainty
```

- correct **and** certain on a high-value class → large **positive**
- wrong **and** certain → large **negative** (the ML actively misleads)
- uncertain (low `c`, high `H̄`) → near **zero** (inspection adds little)

### Class value `v[y]` (derived, not hardcoded)

```
inv[y]  = (N_total / (K_present · n_y))^γ          # inverse frequency, mean-normalised
base[y] = threat_value      if y ∈ threats
        = nonthreat_value    otherwise
v[y]    = clip( base[y] · inv[y] )
```

`v[y]` is **threat-centric** (operational stakes) and **frequency-modulated**
(rarer classes weigh slightly more — the data-derived component). Non-threat
values sit in the **sub-`κ`** region so that a *correctly dismissable*
background/noise signal has `U_ml < κ` and is discarded.

## Reward

Let the **surplus** value of inspecting be `S = U_ml(X) − κ`.

```
R(FORWARD) = S − η · max(0, ρ − ρ*)
R(DISCARD) = − β · max(0, S)
```

### Optimal policy (contextual bandit, γ = 0)

Each signal is independent, so the learned Q-values converge to the conditional
expectations `Q(X,a) = E[ R(a) | X ]`. Ignoring the budget term, the greedy
action prefers FORWARD iff

```
E[S | X] ≥ −β · E[max(S,0) | X]
```

Because the RHS ≤ 0 and equals 0 exactly on the discard region, this reduces to
the clean VoI threshold:

```
FORWARD  ⇔  E[U_ml(X) | X] ≥ κ
```

i.e. **forward iff the expected ML decision value pays for the compute.**

### Role of β (false-negative asymmetry)

`β` does **not** shift the decision threshold (that stays at `U_ml = κ`). It only
scales the *penalty for discarding a signal that was worth forwarding* (`S > 0`),
in proportion to `S`. This is a Neyman-Pearson-style preference: missing a
**confirmable high-value (threat) detection** is far costlier than wasting
compute on a false alarm. Earlier designs that put `−β·U⁺` on the discard branch
were rejected because they lower the threshold to `κ/(1+β)` and cause the agent
to forward almost everything (degenerate "forward-everything" collapse).

### Anti-degeneracy guarantees

1. **Forwarding noise is penalised.** Background/noise has `v[y]` small ⇒
   `U_ml < κ` ⇒ `S < 0` ⇒ `R(FORWARD) = S < 0`, while `R(DISCARD) = 0`. The
   agent strictly prefers discarding obvious background.
2. **Forward-budget regulariser.** `−η·max(0, ρ − ρ*)` adds pressure once the
   running forward rate exceeds the target `ρ*`. `ρ*` is **data-derived**: the
   fraction of signals the cost-aware oracle would forward over a calibration
   pass (`estimate_forward_budget`), so it adapts to the dataset's threat base
   rate instead of being a magic constant.
3. **Curriculum.** Phase 1 fills the replay buffer with oracle demonstrations
   (`teacher_action`), so the student never starts from the trivial
   forward-everything / discard-everything policies.

## Configuration

All coefficients live in `configs/config.yaml → reward:`; none are hardcoded in
code. `v[y]` is computed from the realised training-label counts at run time.

| Key | Symbol | Default | Effect |
|-----|--------|---------|--------|
| `compute_cost` | κ | 0.35 | higher → forward less |
| `miss_aversion` | β | 3.0 | higher → discarding valuable signals hurts more |
| `certainty_blend` | α | 0.5 | confidence vs (1−entropy) mix |
| `value_gamma` | γ | 0.5 | strength of inverse-frequency modulation |
| `threat_value` / `nonthreat_value` | — | 1.5 / 0.25 | operational stakes per group |
| `forward_budget` | ρ* | null→derived | target forward rate |
| `budget_penalty` | η | 1.0 | over-budget pressure |
