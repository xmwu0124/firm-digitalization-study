# IO Structural Estimation - Technical Documentation

## 📖 Overview

This script implements a **dynamic discrete choice model** in the Industrial Organization (IO) tradition, estimating the structural parameters governing firms' digital technology adoption decisions.

---

## 🎯 Model Specification

### Decision Problem

Firms face a **dynamic discrete choice**:
- **Action**: Adopt digital technology (a=1) or wait (a=0)
- **Timing**: Each period, non-adopters decide whether to adopt
- **Irreversibility**: Once adopted, cannot revert
- **Payoffs**: Adoption generates persistent benefits but has fixed costs

### State Variables

**s_t = (size_t, comp_t, tech_t, adopted_t)**

- `size_t`: Log firm size (employment)
- `comp_t`: Industry competition index
- `tech_t`: Tech industry indicator (0/1)
- `adopted_t`: Already adopted indicator (0/1)

### Flow Payoffs

**u(s_t, a_t) = π(s_t, a_t) - FC(a_t)**

**Operating Profit** (if adopted):
```
π(s,1) = α₁·size + α₂·(-comp) + α₃·tech
```

**Costs**:
- Fixed adoption cost: `FC_adopt` (one-time, paid at adoption)
- Maintenance cost: `FC_maintain` (per-period, if technology active)

### Bellman Equation

**V(s) = E_ε [max_a {u(s,a) + ε(a) + β·E[V(s')|s,a]}]**

Where:
- `β`: Discount factor
- `ε(a)`: Type I Extreme Value shocks (scale parameter σ)
- `s' ~ F(s'|s,a)`: State transitions

### State Transitions

**Size evolution** (AR(1) process):
```
log(size_{t+1}) = γ·log(size_t) + η_t,  η_t ~ N(0, σ²)
```

**Competition and industry**: Assumed fixed (can be extended)

**Adoption status**:
```
adopted_{t+1} = max(adopted_t, a_t)
```

---

## 🔬 Estimation Strategy

### Step 1: Discretize State Space

Create finite grids:
- Size: 8 points on [log(size_min), log(size_max)]
- Competition: 5 points on [0, 1]
- Tech: {0, 1}
- Adopted: {0, 1}

**Total states**: 8 × 5 × 2 × 2 = 160 states

### Step 2: Build Transition Matrix

Compute `P[s,a,s']` for all (state, action, next state) triples:
```python
P = build_transition_matrix(state_space, params)
# Shape: (160, 2, 160)
```

### Step 3: Solve Dynamic Program

Use **value function iteration**:

```python
# Initialize
V = zeros(n_states)

# Iterate until convergence
while not converged:
    # Bellman operator
    EV = P @ V  # Expected continuation value
    Q = flow_payoffs + β * EV  # Total value
    V_new = σ * log(sum(exp(Q/σ)))  # Logit inclusive value
    
    # Check convergence
    converged = ||V_new - V|| < tol
    V = V_new

# Extract policy
policy[s,a] = exp(Q[s,a]/σ) / sum_a' exp(Q[s,a']/σ)
```

This gives **choice probabilities** for each state.

### Step 4: Maximum Likelihood Estimation

**Likelihood function**:
```
L(θ | data) = ∏_{i,t} Pr(a_it | s_it; θ)
```

Where `Pr(a|s;θ)` comes from solving the DP with parameters `θ`.

**Objective**:
```
θ̂ = argmax_θ log L(θ | data)
```

**Parameters to estimate**:
```
θ = (α₁, α₂, α₃, FC_adopt, FC_maintain, β, δ, γ, σ_η, σ_ε)
```

### Step 5: Standard Errors

Compute via **numerical Hessian**:
```
SE(θ̂) = sqrt(diag(H(θ̂)⁻¹))
```

Where `H` is the Hessian of the log-likelihood.

---

## 💻 Implementation Details

### JAX Features Used

✅ **JIT compilation**: All core functions decorated with `@jit`
```python
@jit
def bellman_operator(V, P, flow_payoffs, params):
    ...
```

✅ **Vectorized operations**: Use `jnp.einsum` for tensor contractions
```python
EV = jnp.einsum('ijk,k->ij', P, V)
```

✅ **Automatic differentiation**: Can compute gradients if needed
```python
grad_fn = jax.grad(neg_log_likelihood)
```

✅ **64-bit precision**: Essential for numerical stability
```python
jax.config.update("jax_enable_x64", True)
```

### Optimization

**Method**: L-BFGS-B (quasi-Newton with bounds)

**Why**:
- Handles box constraints naturally
- Fast convergence for smooth objectives
- Low memory requirements

**Bounds**:
```python
bounds = [
    (0.0, 2.0),    # α₁ (size benefit)
    (0.0, 2.0),    # α₂ (competition benefit)
    ...
    (0.9, 0.99),   # β (discount factor)
    ...
]
```

### Computational Tricks

1. **Precompute flow payoffs**: Done once per parameter vector
   ```python
   flow_payoffs = compute_flow_payoffs(state_space, params)
   ```

2. **Cache transition matrix**: Only depends on size dynamics
   ```python
   P = build_transition_matrix(state_space, params)
   # Reuse across likelihood evaluations
   ```

3. **Logsumexp for stability**: Avoid numerical overflow
   ```python
   V = scale * logsumexp(Q / scale, axis=1)
   ```

---

## 📊 Output Interpretation

### Parameter Estimates

**Example output**:
```
Parameter Estimates:
------------------------------------------------------------
  alpha_size (benefit of size)          :   0.4523  (SE: 0.0821)
  alpha_comp (benefit of low comp)      :   0.2187  (SE: 0.0654)
  alpha_tech (tech industry premium)    :   0.6742  (SE: 0.1103)
  fc_adopt (fixed cost adoption)        :   2.1456  (SE: 0.3210)
  fc_maintain (maintenance cost)        :   0.1834  (SE: 0.0432)
  beta (discount factor)                :   0.9512  (SE: 0.0089)
  ...
```

**Interpretation**:
- `α₁ = 0.45`: 1% larger firm → 0.45% higher flow payoff if adopted
- `α₃ = 0.67`: Tech firms get 67% higher payoff from adoption
- `FC_adopt = 2.15`: Fixed cost equivalent to 2.15 units of profit
- `β = 0.95`: Firms discount future at ~5% per period

### Counterfactual: Adoption Subsidy

**Policy**: Reduce fixed cost by 1.0 unit

**Results**:
```
Baseline adoption rate: 0.423
With subsidy: 0.518
Increase: 0.095 (22.5%)
```

**Interpretation**: Subsidy of 1.0 increases adoption by 9.5 percentage points (22% relative increase)

---

## 🔬 Methodological Notes

### Comparison to Reduced-Form Methods

| Feature | Reduced-Form (DiD) | Structural (IO) |
|---------|-------------------|-----------------|
| **Causal effect** | ✅ Average treatment effect | ✅ Full distribution of effects |
| **Mechanisms** | ❌ Black box | ✅ Explicit payoffs & dynamics |
| **Counterfactuals** | ❌ Limited | ✅ Any policy scenario |
| **Data requirements** | Low | High |
| **Computation** | Fast | Slow (DP solving) |
| **Assumptions** | Parallel trends | Full model specification |

### When to Use Structural Estimation

**Use when**:
- ✅ Need to evaluate untested policies
- ✅ Want to understand mechanisms
- ✅ Have rich panel data
- ✅ Can credibly specify model

**Don't use when**:
- ❌ Just want causal effect (use DiD)
- ❌ Model is mis-specified
- ❌ Computation is infeasible
- ❌ Data is too sparse

### Extensions

This model can be extended to include:

1. **Heterogeneous effects**:
   ```python
   # Add firm-specific random coefficients
   α₁ᵢ = α₁ + νᵢ,  νᵢ ~ N(0, σ_α²)
   ```

2. **Strategic interactions**:
   ```python
   # Payoff depends on rivals' adoption
   π(s, a, a₋ᵢ) = α₁·size + α₄·(Σ a₋ᵢ)
   ```

3. **Entry/exit**:
   ```python
   # Additional actions: exit market
   a ∈ {0 (wait), 1 (adopt), 2 (exit)}
   ```

4. **Unobserved state variables**:
   ```python
   # Add permanent unobserved type
   s = (size, comp, tech, adopted, type)
   ```

---

## 🚀 Running the Code

### Quick Start

```bash
# Make sure you have JAX installed
pip install jax jaxlib

# Run estimation
python 04_capital_estimation/io_structural_model.py
```

### Expected Runtime

- State space construction: <1 second
- Transition matrix: ~5 seconds
- DP solving (per iteration): ~10 seconds
- Full estimation: ~10-20 minutes (depends on optimization)

### Memory Requirements

- ~50 MB for 160-state model
- Scales as O(n_states²) due to transition matrix

### Troubleshooting

**"DP not converging"**:
- Increase `max_iter` in `solve_dynamic_program`
- Decrease discount factor β slightly
- Check flow payoffs for NaN values

**"Optimization slow"**:
- Reduce state space size (fewer grid points)
- Use better starting values
- Try different optimizer (e.g., Nelder-Mead)

**"Numerical overflow"**:
- Ensure using 64-bit precision
- Use `logsumexp` instead of manual exp/log
- Scale payoffs to reasonable magnitude

---

## 📚 References

### Key Papers

**Dynamic discrete choice**:
- Rust, J. (1987). "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." *Econometrica*, 55(5), 999-1033.

**Conditional choice probabilities**:
- Hotz, V. J., & Miller, R. A. (1993). "Conditional choice probabilities and the estimation of dynamic models." *Review of Economic Studies*, 60(3), 497-529.

**Sequential estimation**:
- Aguirregabiria, V., & Mira, P. (2007). "Sequential estimation of dynamic discrete games." *Econometrica*, 75(1), 1-53.

**Two-step estimation**:
- Arcidiacono, P., & Miller, R. A. (2011). "Conditional choice probability estimation of dynamic discrete choice models with unobserved heterogeneity." *Econometrica*, 79(6), 1823-1867.

### Software

- **JAX**: [https://github.com/google/jax](https://github.com/google/jax)
- **QuantEcon**: [https://quantecon.org/](https://quantecon.org/) - Similar DP examples in Python

---

## ✅ Validation Checklist

Before trusting results:

- [ ] DP converges to fixed point
- [ ] Choice probabilities sum to 1
- [ ] Likelihood is finite and improving
- [ ] Parameter estimates are economically reasonable
- [ ] Standard errors are not too large
- [ ] Counterfactuals make sense
- [ ] Robustness to starting values
- [ ] Robustness to state space discretization

---

## 🎓 Teaching Notes

This script is designed to be pedagogical:

1. **Clear structure**: Each step (payoffs → transitions → DP → likelihood → estimation) is separate

2. **Extensive comments**: Explains *why* not just *what*

3. **Named tuples**: `ModelParams` and `StateSpace` make code readable

4. **Logging**: Tracks progress and aids debugging

5. **Modular**: Easy to swap out components (e.g., different transition spec)

**Use in class**:
- Walk through model specification on board
- Show DP solution convergence
- Discuss identification (which parameters come from what variation)
- Compare structural vs reduced-form results

---

**Questions?** See main README.md or contact the research team.
